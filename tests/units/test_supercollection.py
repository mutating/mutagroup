import sys
from collections import Counter
from dataclasses import astuple, replace
from pathlib import Path
from subprocess import run
from textwrap import dedent
from typing import (
    Any,
    Collection,
    Counter as CounterType,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import pytest
from cstvis import Changer, Converter, Coordinate
from cstvis.visitors.comments_aggregator import CommentsAggregator
from libcst import (
    Add,
    Assign,
    BinaryOperation,
    Comment,
    CSTNode,
    CSTTransformer,
    CSTVisitor,
    Divide,
    FlattenSentinel,
    Float,
    Integer,
    Module,
    Multiply,
    ParserSyntaxError,
    RemovalSentinel,
    Subtract,
)
from libcst._nodes.internal import CodegenState
from libcst._parser import entrypoints
from libcst.metadata import MetadataWrapper, ProviderT
from metacode import ParsedComment

from mutagroup import Context, MutationsCollection, MutationsSuperCollection


@pytest.mark.parametrize(
    ('inline_comment', 'expected_coordinates'),
    [
        pytest.param('', [(1, 'J-1'), (2, 'I-1'), (2, 'J-1'), (2, 'J-2')], id='without-comment'),
        pytest.param('  # muta: ignore[J-1]', [(1, 'J-1'), (2, 'I-1'), (2, 'J-2')], id='with-mute'),
    ],
)
def test_combined_discovery_walks_once_and_application_reuses_prepared_source(
    inline_comment: str,
    expected_coordinates: List[Tuple[int, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Find candidates in one discovery walk and reuse the prepared source to apply them.

    Count parses at the entry point shared by import aliases. In this source, a full parse
    produces two statements and the metacode fragment produces one. Track visits to both
    assignments, both integers, and the addition operator to catch walks within an assignment.
    Allow one transformation walk per coordinate.
    Compare metadata visits and directive parses with a single-pack run to allow one-time preparation.
    """
    filtered_pack = MutationsCollection('I')
    unfiltered_pack = MutationsCollection('J')

    def keep(node: Integer) -> Integer:
        return node

    filtered_pack.mutation(keep)
    unfiltered_pack.mutation(keep)

    @unfiltered_pack.mutation
    def subtract(node: Add) -> Subtract:
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    @filtered_pack.filter
    def keep_second_value(node: Integer) -> bool:
        return node.value == '2'

    counts: CounterType[str] = Counter()
    metadata_depth = 0
    original_parse = entrypoints._parse
    original_resolve = MetadataWrapper.resolve_many
    original_visit = CSTNode.visit

    def record_parse(*args: Any, **kwargs: Any) -> CSTNode:
        parsed = original_parse(*args, **kwargs)
        if isinstance(parsed, Module):
            counts['source_parses' if len(parsed.body) > 1 else 'directive_parses'] += 1
        return parsed

    def resolve_metadata(
        wrapper: MetadataWrapper,
        providers: Collection[ProviderT],
    ) -> Mapping[ProviderT, Mapping[CSTNode, object]]:
        nonlocal metadata_depth
        metadata_depth += 1
        try:
            return original_resolve(wrapper, providers)
        finally:
            metadata_depth -= 1

    def record_visit(
        node: CSTNode,
        visitor: Union[CSTVisitor, CSTTransformer],
    ) -> Union[CSTNode, RemovalSentinel, FlattenSentinel[CSTNode]]:
        if isinstance(node, (Assign, Integer, Add)):
            counts['metadata_visits' if metadata_depth else 'node_visits'] += 1
        return original_visit(node, visitor)

    monkeypatch.setattr(entrypoints, '_parse', record_parse)
    monkeypatch.setattr(MetadataWrapper, 'resolve_many', resolve_metadata)
    monkeypatch.setattr(CSTNode, 'visit', record_visit)

    baseline_source = f'a = 1\nb = 2 + value{inline_comment}'
    baseline = MutationsSuperCollection([filtered_pack]).get_changer(baseline_source)
    (baseline_coordinate,) = baseline.iterate_coordinates()
    baseline_discovery_metadata_visits = counts['metadata_visits']
    assert baseline.apply_coordinate(baseline_coordinate) == baseline_source
    baseline_total_metadata_visits = counts['metadata_visits']
    baseline_directive_parses = counts['directive_parses']
    counts.clear()

    source = f'x = 1\ny = 2 + value{inline_comment}'
    changer = MutationsSuperCollection([filtered_pack, unfiltered_pack]).get_changer(source)
    coordinates = list(changer.iterate_coordinates())
    assert Counter((item.start_line, item.converter_id) for item in coordinates) == Counter(expected_coordinates)
    assert counts['source_parses'] == 1
    assert counts['node_visits'] == 5
    assert counts['metadata_visits'] <= baseline_discovery_metadata_visits
    assert [changer.apply_coordinate(coordinate) for coordinate in coordinates] == [
        source.replace('2 + value', '2 - value') if coordinate.converter_id == 'J-2' else source for coordinate in coordinates
    ]
    assert counts['source_parses'] == 1
    assert counts['metadata_visits'] <= baseline_total_metadata_visits
    assert counts['node_visits'] <= 5 * (1 + len(coordinates))
    assert counts['directive_parses'] <= baseline_directive_parses


def test_offset_queries_share_code_generation_across_packs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse source-offset preparation across packs, repeated searches, and applications.

    Count code-generation passes during setup and discovery on the same source. The single-pack
    run reads one span, while the combined run reads two distinct spans. When applying coordinates,
    allow one generation pass to render each mutant and one-time preparation on a fresh changer.
    """
    first = MutationsCollection('I')
    second = MutationsCollection('J')
    spans: List[Tuple[int, int]] = []
    codegen_walks = 0
    original_codegen = Module._codegen

    def record_codegen(module: Module, state: CodegenState, **options: Any) -> None:
        nonlocal codegen_walks
        codegen_walks += 1
        original_codegen(module, state, **options)

    @first.mutation
    def keep_integer(node: Integer) -> Integer:
        return node

    @second.mutation
    def replace_number(node: Union[Integer, Float], context: Context) -> Integer:
        position = context.position
        assert position.source[position.start_offset:position.end_offset] == node.value
        return Integer('0')

    @first.filter
    @second.filter
    def check_source_slice(node: Union[Integer, Float], context: Context) -> bool:
        position = context.position
        start, end = position.start_offset, position.end_offset
        spans.append((start, end))
        return position.source[start:end] == node.value

    monkeypatch.setattr(Module, '_codegen', record_codegen)
    source = 'x = 1\ny = 2.5'
    baseline = MutationsSuperCollection([first]).get_changer(source)
    assert [coordinate.converter_id for coordinate in baseline.iterate_coordinates()] == ['I-1']
    assert spans == [(4, 5)]
    baseline_codegen_walks = codegen_walks
    spans.clear()
    codegen_walks = 0

    packs = MutationsSuperCollection([first, second])
    changer = packs.get_changer(source)
    coordinates = list(changer.iterate_coordinates())
    assert Counter(coordinate.converter_id for coordinate in coordinates) == Counter(['I-1', 'J-1', 'J-1'])
    assert Counter(spans) == Counter([(4, 5), (4, 5), (10, 13)])
    assert codegen_walks <= baseline_codegen_walks
    discovery_codegen_walks = codegen_walks
    assert list(changer.iterate_coordinates()) == coordinates
    assert codegen_walks == discovery_codegen_walks
    expected_number_mutants = {1: 'x = 0\ny = 2.5', 2: 'x = 1\ny = 0'}
    assert [changer.apply_coordinate(coordinate) for coordinate in coordinates] == [
        expected_number_mutants[coordinate.start_line] if coordinate.converter_id == 'J-1' else source
        for coordinate in coordinates
    ]
    assert codegen_walks - discovery_codegen_walks <= len(coordinates)

    fresh_changer = packs.get_changer(source)
    first_coordinate, second_coordinate = (coordinate for coordinate in coordinates if coordinate.converter_id == 'J-1')
    assert fresh_changer.apply_coordinate(first_coordinate) == expected_number_mutants[first_coordinate.start_line]
    codegen_walks_after_first = codegen_walks
    assert fresh_changer.apply_coordinate(second_coordinate) == expected_number_mutants[second_coordinate.start_line]
    assert codegen_walks - codegen_walks_after_first <= 1


def test_combined_coordinates_apply_independently_by_mutation_id_and_node_span() -> None:
    """Apply the selected mutation at its span without rerunning discovery filters.

    Share callbacks across packs with distinct defaults and verify each registration's metadata.
    Use overlapping annotations to check that each matching filter registration runs once.
    Register I-1 before I-13 and try J-1 against J-13 to check exact selection.
    Add a file path to each coordinate as an engine might do when collecting results.
    """
    arithmetic = MutationsCollection('I', meta={'pack': 'I'})
    alternate = MutationsCollection('J', meta={'pack': 'J'})
    applications: List[Tuple[str, int, int]] = []
    operators: Dict[str, Type[Union[Subtract, Divide, Multiply]]] = {'I-1': Divide, 'I-13': Subtract, 'J-13': Multiply}

    def change_operator(node: Add, context: Context) -> Union[Subtract, Divide, Multiply]:
        assert context.meta is not None
        mutation_id = context.meta['mutation_id']
        assert context.meta['pack'] == mutation_id.partition('-')[0]
        assert context.meta['mutation_name'] == 'change_operator'
        position = context.position.coordinate
        applications.append((mutation_id, position.start_line, position.start_column))
        operator = operators[mutation_id]
        return operator(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    divide = arithmetic.mutation(1)(change_operator)
    subtract = arithmetic.mutation(13)(change_operator)
    alternate.mutation(13)(change_operator)
    pack_filter_calls: List[Tuple[str, int, int]] = []
    bound_filter_calls: List[Tuple[str, int, int]] = []

    @arithmetic.filter
    @alternate.filter
    def allow_operator(node: Union[Add, CSTNode], ctx: Context) -> bool:
        assert ctx.meta is not None
        prefix = ctx.meta['pack']
        position = ctx.position.coordinate
        pack_filter_calls.append((prefix, position.start_line, position.start_column))
        assert isinstance(node, Add)
        return not node.whitespace_before.empty if prefix == 'I' else position.start_line != 2

    @subtract.filter
    @divide.filter
    def restrict_subtraction_to_first_line(_node: Union[Add, CSTNode], ctx: Context) -> bool:
        assert ctx.meta is not None
        assert ctx.meta['mutation_name'] == 'change_operator'
        mutation_id = ctx.meta['mutation_id']
        position = ctx.position.coordinate
        bound_filter_calls.append((mutation_id, position.start_line, position.start_column))
        return mutation_id == 'I-1' or position.start_line == 1

    source = 'π = 1+2 + 3  # keep\r\nother = 4 + 5'
    changer = MutationsSuperCollection([arithmetic, alternate]).get_changer(source)
    coordinates = list(changer.iterate_coordinates())
    assert applications == []
    expected_pack_calls = Counter(
        (prefix, line, column) for prefix in ('I', 'J') for line, column in [(1, 5), (1, 8), (2, 10)]
    )
    expected_bound_calls = Counter([('I-1', 1, 8), ('I-13', 1, 8), ('I-1', 2, 10), ('I-13', 2, 10)])
    assert Counter(pack_filter_calls) == expected_pack_calls
    assert Counter(bound_filter_calls) == expected_bound_calls

    expected_mutants = {
        (None, 'Add', 1, 5, 1, 6, 'J-13'): 'π = 1*2 + 3  # keep\r\nother = 4 + 5',
        (None, 'Add', 1, 8, 1, 9, 'I-13'): 'π = 1+2 - 3  # keep\r\nother = 4 + 5',
        (None, 'Add', 1, 8, 1, 9, 'I-1'): 'π = 1+2 / 3  # keep\r\nother = 4 + 5',
        (None, 'Add', 1, 8, 1, 9, 'J-13'): 'π = 1+2 * 3  # keep\r\nother = 4 + 5',
        (None, 'Add', 2, 10, 2, 11, 'I-1'): 'π = 1+2 + 3  # keep\r\nother = 4 / 5',
    }
    assert len(coordinates) == len(expected_mutants)
    assert {astuple(coordinate) for coordinate in coordinates} == expected_mutants.keys()
    positions = [(coordinate.start_line, coordinate.start_column) for coordinate in coordinates]
    assert positions == sorted(positions)

    selections = [*reversed(coordinates), coordinates[0]]
    assert [changer.apply_coordinate(replace(coordinate, file=Path('module.py'))) for coordinate in selections] == [
        expected_mutants[astuple(coordinate)] for coordinate in selections
    ]
    assert changer.apply_coordinate(replace(coordinates[0], converter_id='J-1')) == source
    assert applications == [
        (coordinate.converter_id, coordinate.start_line, coordinate.start_column) for coordinate in selections
    ]
    assert Counter(pack_filter_calls) == expected_pack_calls
    assert Counter(bound_filter_calls) == expected_bound_calls


@pytest.mark.parametrize(
    ('source', 'expected_spans', 'expected_mutants'),
    [
        pytest.param(
            'x = 1 + 2 + 3', [(1, 4, 1, 9), (1, 4, 1, 13)], ['x = 0 + 3', 'x = 0'], id='different-end-columns',
        ),
        pytest.param(
            'x = (1 + 2\n       + 3)\n', [(1, 5, 1, 10), (1, 5, 2, 10)], ['x = (0\n       + 3)\n', 'x = 0\n'], id='different-end-lines',
        ),
        pytest.param(
            'x = 1 ** 2 ** 3\n', [(1, 4, 1, 15), (1, 9, 1, 15)], ['x = 0\n', 'x = 1 ** 0\n'], id='different-start-columns',
        ),
        pytest.param(
            'x = (1 **\n     2 ** 3)\n', [(1, 5, 2, 11), (2, 5, 2, 11)], ['x = 0\n', 'x = (1 **\n     0)\n'], id='different-start-lines',
        ),
    ],
)
def test_nested_coordinates_use_the_full_node_span(
    source: str,
    expected_spans: List[Tuple[int, int, int, int]],
    expected_mutants: List[str],
) -> None:
    """Discover and apply both nested nodes when only one position field differs.

    Addition groups from the left so the nodes share a start. Exponentiation groups
    from the right so they share an end.
    """
    pack = MutationsCollection('I')

    @pack.mutation
    def replace(_node: BinaryOperation) -> Integer:
        return Integer('0')

    changer = MutationsSuperCollection([pack]).get_changer(source)
    coordinates = sorted(
        changer.iterate_coordinates(),
        key=lambda coordinate: (coordinate.start_line, coordinate.start_column, coordinate.end_line, coordinate.end_column),
    )
    assert coordinates == [
        Coordinate(None, 'BinaryOperation', *span, 'I-1') for span in expected_spans
    ]
    assert [changer.apply_coordinate(coordinate) for coordinate in coordinates] == expected_mutants


@pytest.mark.parametrize('muted_target', [None, 'W', 'W-1'], ids=['unmuted', 'pack-prefix', 'mutation-id'])
def test_concrete_and_wildcard_mutations_respect_inline_mutes(muted_target: Optional[str]) -> None:
    """Discover both matches on one node and suppress the wildcard by ID or pack prefix.

    Give the surviving handler the full comment so it can read a separate metacode on that line.
    """
    concrete = MutationsCollection('C')
    wildcard = MutationsCollection('W')
    comment = f' muta: ignore[{muted_target}] # hint: note["x"]' if muted_target is not None else None
    suffix = f'  #{comment}' if comment is not None else ''

    @concrete.mutation
    def set_two(node: Integer, context: Context) -> Integer:
        assert context.meta is not None
        assert context.meta['mutation_name'] == 'set_two'
        assert context.comment == comment
        expected_hints = [ParsedComment('hint', 'note', ['x'])] if comment is not None else []
        assert context.get_metacodes('hint') == expected_hints
        return node.with_changes(value='2')

    @wildcard.mutation
    def set_three(node: CSTNode, context: Context) -> Integer:
        assert context.meta is not None
        assert context.meta['mutation_name'] == 'set_three'
        assert isinstance(node, Integer)
        return node.with_changes(value='3')

    @wildcard.filter
    def only_integers(node: CSTNode) -> bool:
        return isinstance(node, Integer)

    changer = MutationsSuperCollection([concrete, wildcard]).get_changer(f'x = 1{suffix}')
    expected_mutants = [('C-1', f'x = 2{suffix}')]
    if muted_target is None:
        expected_mutants.append(('W-1', 'x = 3'))
    assert Counter(
        (coordinate.converter_id, changer.apply_coordinate(coordinate)) for coordinate in changer.iterate_coordinates()
    ) == Counter(expected_mutants)


@pytest.mark.parametrize('filter_scope', ['pack', 'bound', 'native', 'line-mute', 'project-mute'])
def test_rejected_parent_keeps_child_candidates_from_other_packs(filter_scope: str) -> None:
    """Keep discovering child candidates when a filter or mute rejects the parent node."""
    parents = MutationsCollection('P')
    leaves = MutationsCollection('L')
    empty = MutationsCollection('E')
    rejected: List[str] = []

    @parents.mutation
    def keep(node: BinaryOperation) -> BinaryOperation:
        return node

    @leaves.mutation
    def replace(node: Integer) -> Integer:
        return node.with_changes(value='0')

    @parents.filter
    def foreign_type(_node: Integer) -> bool:
        pytest.fail('A pack filter ran where only another pack had a candidate.')

    @empty.filter
    def no_candidates(_node: CSTNode) -> bool:
        pytest.fail('An empty pack ran a filter.')

    def veto(node: BinaryOperation) -> bool:
        rejected.append(type(node).__name__)
        return False

    if filter_scope == 'pack':
        parents.filter(veto)
    elif filter_scope == 'bound':
        keep.filter(veto)

    source = 'x = 1 + 2  # muta: ignore[P]' if filter_scope == 'line-mute' else 'x = 1 + 2'
    ignored = ['P-1'] if filter_scope == 'project-mute' else []
    changer = MutationsSuperCollection([parents, leaves, empty]).get_changer(source, ignored=ignored)
    if filter_scope == 'native':
        changer.filter(veto)  # type: ignore[arg-type]

    coordinates = list(changer.iterate_coordinates())
    expected_rejections = ['BinaryOperation'] if filter_scope in ('pack', 'bound', 'native') else []
    assert rejected == expected_rejections
    assert coordinates == [
        Coordinate(None, 'Integer', 1, 4, 1, 5, 'L-1'),
        Coordinate(None, 'Integer', 1, 8, 1, 9, 'L-1'),
    ]
    assert [changer.apply_coordinate(coordinate) for coordinate in coordinates] == [
        source.replace('1 + 2', '0 + 2'), source.replace('1 + 2', '1 + 0'),
    ]


@pytest.mark.parametrize(
    'source',
    [
        pytest.param(
            'x = 1  # Mutator_2: ignore[J] # muta: ignore[I-13]\ny = 2  # Mutator_2: ignore[I-13]\n',
            id='canonical-keys',
        ),
        pytest.param(
            'x = 1  # type: typing.List[int] # Mutator_2: ignore[J] # muta: ignore[I-13]\ny = 2  # Mutator_2: ignore[I-13]\n',
            id='keys-after-qualified-type-comment',
        ),
        pytest.param(
            'x = 1  # Mutator_2 ignore J-13 # muta ignore I-13\ny = 2  # Mutator_2 ignore I-13\n',
            id='legacy-keys',
        ),
    ],
)
def test_configured_comment_key_is_scoped_to_each_run(source: str) -> None:
    """Let each run use its own metacode key even when it shares packs and comment text."""
    first = MutationsCollection('I')
    second = MutationsCollection('J')

    def keep(node: Integer) -> Integer:
        return node

    first.mutation(13)(keep)
    first.mutation(2)(keep)
    second.mutation(13)(keep)
    packs = [first, second]

    custom = MutationsSuperCollection(packs, metacode_key='Mutator_2').get_changer(source)
    default = MutationsSuperCollection(packs).get_changer(source)
    expected_custom = Counter([(1, 'I-13'), (1, 'I-2'), (2, 'I-2'), (2, 'J-13')])
    assert Counter((item.start_line, item.converter_id) for item in custom.iterate_coordinates()) == expected_custom
    assert Counter((item.start_line, item.converter_id) for item in default.iterate_coordinates()) == Counter([
        (1, 'I-2'), (1, 'J-13'), (2, 'I-13'), (2, 'I-2'), (2, 'J-13'),
    ])
    assert Counter((item.start_line, item.converter_id) for item in custom.iterate_coordinates()) == expected_custom


def test_line_mutes_and_project_mutes_compose_across_packs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply exact mutes across packs and to all statements that share a line inside a function.

    Reuse the collected comments when applying a coordinate inside the function and repeating discovery.
    """
    first = MutationsCollection('I')
    second = MutationsCollection('II')
    third = MutationsCollection('J')

    def subtract(node: Add) -> Subtract:
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    first.mutation(13)(subtract)
    first.mutation(subtract)
    second.mutation(13)(subtract)
    second.mutation(subtract)
    third.mutation(13)(subtract)
    third.mutation(130)(subtract)

    comment_visits = 0
    original_visit = CommentsAggregator.on_visit

    def count_comment(aggregator: CommentsAggregator, node: CSTNode) -> bool:
        nonlocal comment_visits
        if isinstance(node, Comment):
            comment_visits += 1
        return original_visit(aggregator, node)

    monkeypatch.setattr(CommentsAggregator, 'on_visit', count_comment)
    packs = MutationsSuperCollection([first, second, third])
    source = 'def compute():\n    x = 1 + 2 + 3; z = 4 + 5  # muta: ignore[I-13, II]\ny = 4 + 5'
    changer = packs.get_changer(source, ignored=iter(['J-13', 'J-13', 'J-1300', 'I', 'unknown']))
    coordinates = list(changer.iterate_coordinates())
    assert Counter((item.start_line, item.start_column, item.converter_id) for item in coordinates) == Counter([
        (2, 10, 'I-1'), (2, 10, 'J-130'),
        (2, 14, 'I-1'), (2, 14, 'J-130'),
        (2, 25, 'I-1'), (2, 25, 'J-130'),
        (3, 6, 'I-13'), (3, 6, 'I-1'), (3, 6, 'II-13'), (3, 6, 'II-1'), (3, 6, 'J-130'),
    ])
    inner_coordinate = next(item for item in coordinates if (item.start_line, item.start_column, item.converter_id) == (2, 10, 'I-1'))
    assert changer.apply_coordinate(inner_coordinate) == source.replace('1 + 2', '1 - 2')
    assert list(changer.iterate_coordinates()) == coordinates
    assert comment_visits <= 1


def test_one_shot_project_mutes_reach_every_pack() -> None:
    """Use every ID from a one-shot iterable across packs and leave the next run independent."""
    first = MutationsCollection('I')
    second = MutationsCollection('J')

    def keep(node: Integer) -> Integer:
        return node

    first.mutation(keep)
    second.mutation(keep)
    packs = MutationsSuperCollection(iter([first, second]))
    assert list(packs.get_changer('1', ignored=iter(['I-1', 'J-1'])).iterate_coordinates()) == []
    assert Counter(coordinate.converter_id for coordinate in packs.get_changer('1').iterate_coordinates()) == Counter(['I-1', 'J-1'])


def test_supercollection_can_be_reused_with_independent_sources_and_ignored_lists() -> None:
    """Keep source offsets, comments, and project mutes local to each changer.

    Apply a known coordinate before discovery and use distinct results to identify its registration.
    """
    pack = MutationsCollection('I')

    def replace(node: Integer, context: Context) -> Integer:
        position = context.position
        assert position.source[position.start_offset:position.end_offset] == node.value
        assert context.comment == (' keep' if node.value == '3' else None)
        assert context.meta is not None
        return node.with_changes(value=context.meta['value'])

    pack.mutation(replace, meta={'value': '7'})
    pack.mutation(replace, meta={'value': '8'})
    packs = MutationsSuperCollection([pack])
    ignored = ['I-2']
    first = packs.get_changer('x = 1  # muta: ignore[I-1]\ny = 2', ignored=ignored)
    ignored.clear()
    second = packs.get_changer('long_name = 3  # keep', ignored=ignored)

    (old_target,) = first.iterate_coordinates()
    assert old_target == Coordinate(None, 'Integer', 2, 4, 2, 5, 'I-1')
    assert first.apply_coordinate(old_target) == 'x = 1  # muta: ignore[I-1]\ny = 7'
    assert second.apply_coordinate(Coordinate(None, 'Integer', 1, 12, 1, 13, 'I-2')) == 'long_name = 8  # keep'
    new_targets = list(second.iterate_coordinates())
    assert Counter(map(astuple, new_targets)) == Counter([
        (None, 'Integer', 1, 12, 1, 13, 'I-1'),
        (None, 'Integer', 1, 12, 1, 13, 'I-2'),
    ])
    assert Counter((item.converter_id, second.apply_coordinate(item)) for item in new_targets) == Counter([
        ('I-1', 'long_name = 7  # keep'),
        ('I-2', 'long_name = 8  # keep'),
    ])
    assert list(first.iterate_coordinates()) == [old_target]


@pytest.mark.parametrize('filter_scope', ['pack', 'bound', 'native'])
def test_each_discovery_reevaluates_stateful_filters(filter_scope: str) -> None:
    """Reevaluate filters on each discovery so changes in their state take effect.

    Put the same literal on two lines to make the filter decide by position.
    """
    pack = MutationsCollection('I')
    allowed_line = 1
    seen: List[int] = []

    @pack.mutation
    def replace(node: Integer) -> Integer:
        return node.with_changes(value='0')

    def allow_line(_node: Integer, context: Context) -> bool:
        line = context.position.coordinate.start_line
        seen.append(line)
        return line == allowed_line

    if filter_scope == 'pack':
        pack.filter(allow_line)
    elif filter_scope == 'bound':
        replace.filter(allow_line)
    changer = MutationsSuperCollection([pack]).get_changer('x = 1\ny = 1')
    if filter_scope == 'native':
        changer.filter(allow_line)  # type: ignore[arg-type]
    assert [item.start_line for item in changer.iterate_coordinates()] == [1]
    allowed_line = 2
    assert [item.start_line for item in changer.iterate_coordinates()] == [2]
    assert seen == [1, 2, 1, 2]


def test_combined_discovery_agrees_with_separate_runs_for_distinct_node_types() -> None:
    """Combine different node types in source order and preserve the mutants from separate runs."""
    operators = MutationsCollection('O')
    numbers = MutationsCollection('N')

    @operators.mutation
    def subtract(node: Add) -> Subtract:
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    @numbers.mutation
    def replace(node: Integer) -> Integer:
        return node.with_changes(value='0')

    @numbers.filter
    def keep_odd(node: Integer) -> bool:
        return int(node.value) % 2 == 1

    source = 'x = 1 + 2  # muta: ignore[O]\ny = 3 + 5'
    packs = [operators, numbers]
    standalone = [Changer(source, collector=pack.collector()) for pack in packs]
    combined = MutationsSuperCollection(packs).get_changer(source)
    targets = list(combined.iterate_coordinates())
    expected = {
        astuple(coordinate): changer.apply_coordinate(coordinate)
        for changer in standalone
        for coordinate in changer.iterate_coordinates()
    }
    assert [(coordinate.start_line, coordinate.start_column) for coordinate in targets] == [(1, 4), (2, 4), (2, 6), (2, 8)]
    assert len(expected) == 4
    assert {astuple(coordinate): combined.apply_coordinate(coordinate) for coordinate in targets} == expected


@pytest.mark.parametrize(
    ('node_annotation', 'keeps_integer'),
    [
        pytest.param(Add, True, id='concrete-type'),
        pytest.param(CSTNode, False, id='all-types'),
        pytest.param(Union[Add, CSTNode], False, id='overlapping-wildcard'),
    ],
)
def test_native_and_pack_filters_respect_candidate_scope(node_annotation: object, keeps_integer: bool) -> None:
    """Run each changer filter at most once per matching node and stop at the first rejection.

    Register native callbacks after the first discovery and check that they stay local to that changer.
    Only nodes allowed by the changer filter reach the pack and bound filters.
    """
    pack = MutationsCollection('I')

    @pack.mutation
    def subtract(node: Add) -> Subtract:
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    @pack.mutation
    def replace_integer(node: Integer) -> Integer:
        return node.with_changes(value='0')

    pack_filter_lines: List[int] = []

    @pack.filter
    def record_filter_line(_node: Add, context: Context) -> bool:
        assert context.meta is None
        pack_filter_lines.append(context.position.coordinate.start_line)
        return True

    @pack.filter(meta={})
    def skip_second_line(_node: Add, context: Context) -> bool:
        assert context.meta == {}
        return context.position.coordinate.start_line != 2

    source = 'a = 1 + b\nc = d + e\nf = g + h\ni = j + 8.5\n'
    packs = MutationsSuperCollection([pack])
    changer = packs.get_changer(source)
    initial_coordinates = list(changer.iterate_coordinates())
    assert Counter((coordinate.class_name, coordinate.start_line) for coordinate in initial_coordinates) == Counter([
        ('Integer', 1), ('Add', 1), ('Add', 3), ('Add', 4),
    ])
    pack_filter_lines.clear()

    def multiply(node: Add) -> Multiply:
        return Multiply(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    native_converter = changer.converter(multiply)  # type: ignore[arg-type]
    assert isinstance(native_converter, Converter)

    @changer.converter  # type: ignore[arg-type]
    def zero_float(node: Float) -> Float:
        return node.with_changes(value='0.0')

    native_bound_filter_lines: List[int] = []

    @native_converter.filter  # type: ignore[arg-type]
    def skip_fourth_line(_node: Add, context: Context) -> bool:
        assert context.meta is None
        line = context.position.coordinate.start_line
        native_bound_filter_lines.append(line)
        return line != 4

    changer_filter_calls: List[Tuple[str, int, int]] = []

    def skip_configured_line(node: CSTNode, context: Context) -> bool:
        assert context.meta is not None
        line = context.position.coordinate.start_line
        skip_line = context.meta['skip_line']
        changer_filter_calls.append((type(node).__name__, line, skip_line))
        return line != skip_line

    skip_configured_line.__annotations__['node'] = node_annotation
    changer.filter(skip_configured_line, meta={'skip_line': 1})
    changer.filter(skip_configured_line, meta={'skip_line': 5})

    expected_mutants = Counter([
        (2, source.replace('d + e', 'd * e')),
        (3, source.replace('g + h', 'g - h')),
        (3, source.replace('g + h', 'g * h')),
        (4, source.replace('j + 8.5', 'j - 8.5')),
        (4, source.replace('8.5', '0.0')),
    ])
    expected_filter_calls = Counter(('Add', line, 1) for line in range(1, 5))
    expected_filter_calls.update(('Add', line, 5) for line in range(2, 5))
    if keeps_integer:
        expected_mutants[(1, source.replace('1 + b', '0 + b'))] += 1
    else:
        expected_filter_calls[('Integer', 1, 1)] += 1
        expected_filter_calls.update([('Float', 4, 1), ('Float', 4, 5)])
    assert Counter(
        (coordinate.start_line, changer.apply_coordinate(coordinate)) for coordinate in changer.iterate_coordinates()
    ) == expected_mutants
    assert Counter(changer_filter_calls) == expected_filter_calls
    assert pack_filter_lines == native_bound_filter_lines == [2, 3, 4]
    assert list(packs.get_changer(source).iterate_coordinates()) == initial_coordinates


def test_native_converter_ids_remain_stable_after_filtering() -> None:
    """Keep IDs and metadata distinct when one of two native registrations is filtered out."""
    pack = MutationsCollection('I')

    @pack.mutation
    def set_four(node: Integer) -> Integer:
        return node.with_changes(value='4')

    def replace_value(node: Integer, context: Context) -> Integer:
        assert context.meta is not None
        return node.with_changes(value=context.meta['value'])

    changer = MutationsSuperCollection([pack]).get_changer('x = 1')
    first_registration = changer.converter(replace_value, meta={'value': '2'})  # type: ignore[arg-type]
    assert isinstance(first_registration, Converter)
    changer.converter(replace_value, meta={'value': '3'})  # type: ignore[arg-type]

    coordinates = list(changer.iterate_coordinates())
    assert len({coordinate.converter_id for coordinate in coordinates}) == len(coordinates) == 3
    ids_by_mutant = {changer.apply_coordinate(coordinate): coordinate.converter_id for coordinate in coordinates}
    assert ids_by_mutant.keys() == {'x = 2', 'x = 3', 'x = 4'}

    @first_registration.filter  # type: ignore[arg-type]
    def reject(_node: Integer) -> bool:
        return False

    assert Counter(
        (changer.apply_coordinate(coordinate), coordinate.converter_id) for coordinate in changer.iterate_coordinates()
    ) == Counter((mutant, ids_by_mutant[mutant]) for mutant in ('x = 3', 'x = 4'))


def test_empty_supercollection_accepts_native_registrations() -> None:
    """Start with no candidates and let a caller add a native converter to the changer."""
    changer = MutationsSuperCollection([]).get_changer('answer = 42')
    assert list(changer.iterate_coordinates()) == []

    @changer.converter  # type: ignore[arg-type]
    def replace(node: Integer) -> Integer:
        return node.with_changes(value='0')

    (coordinate,) = changer.iterate_coordinates()
    assert changer.apply_coordinate(coordinate) == 'answer = 0'


@pytest.mark.parametrize('source', ['', '# just a comment\n'], ids=['empty', 'comment-only'])
def test_pack_with_no_matching_nodes_has_no_coordinates(source: str) -> None:
    """Keep a valid run empty when the source contains no target node type."""
    pack = MutationsCollection('I')

    @pack.mutation
    def replace(_node: Integer) -> Integer:
        pytest.fail('A mutation ran without a matching node.')

    assert list(MutationsSuperCollection([pack]).get_changer(source).iterate_coordinates()) == []


def test_invalid_source_surfaces_parse_error() -> None:
    """Surface LibCST's parse error when a mutation run receives invalid Python source."""
    pack = MutationsCollection('I')

    @pack.mutation
    def keep(node: Integer) -> Integer:
        return node

    packs = MutationsSuperCollection([pack])
    with pytest.raises(ParserSyntaxError):
        list(packs.get_changer('x = (').iterate_coordinates())


@pytest.mark.parametrize('reuse_pack', [False, True], ids=['same-prefix', 'same-instance'])
def test_supercollection_rejects_duplicate_prefixes(reuse_pack: bool) -> None:
    """Keep pack-level muting and ID routing unambiguous when assembling the run."""
    pack = MutationsCollection('I')
    other = pack if reuse_pack else MutationsCollection('I')

    with pytest.raises(ValueError, match="prefix 'I'"):
        MutationsSuperCollection([pack, other])


@pytest.mark.parametrize('key', ['', 'not-a-key', 'class'])
def test_supercollection_rejects_invalid_metacode_keys(key: str) -> None:
    """Reject keys that cannot be parsed as metacode identifiers."""
    with pytest.raises(ValueError, match='metacode key'):
        MutationsSuperCollection([], metacode_key=key)


def _write_pack_plugin(directory: Path, name: str, prefix: str, number: int, replacement: str) -> None:
    distribution = directory / f'{name}-1.0.dist-info'
    distribution.mkdir()
    (distribution / 'METADATA').write_text(
        f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n',
        encoding='utf-8',
    )
    (distribution / 'entry_points.txt').write_text(f'[pristan]\n{name} = {name}\n', encoding='utf-8')
    (directory / f'{name}.py').write_text(
        dedent(f"""
            from libcst import Integer

            from mutagroup import MutationsCollection, mutations

            pack = MutationsCollection({prefix!r})

            @pack.mutation({number})
            def replace(node: Integer) -> Integer:
                return node.with_changes(value={replacement!r})

            @mutations.plugin({name!r})
            def provide_pack() -> MutationsCollection:
                return pack
        """),
        encoding='utf-8',
    )


def _run_with_plugins(directory: Path, script: str) -> None:
    setup = dedent("""
        import sys

        sys.path.insert(0, sys.argv[1])

        from mutagroup import MutationsSuperCollection
    """)
    result = run(
        [sys.executable, '-c', setup + dedent(script), str(directory)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def two_pack_plugins(tmp_path: Path) -> Path:
    _write_pack_plugin(tmp_path, 'mutagroup_test_first_pack', 'I', 13, '42')
    _write_pack_plugin(tmp_path, 'mutagroup_test_second_pack', 'J', 7, '99')
    return tmp_path


def test_get_mutations_loads_and_applies_two_entry_point_plugins(two_pack_plugins: Path) -> None:
    """Discover both plugin modules on demand and apply their mutations independently."""
    _run_with_plugins(two_pack_plugins, """
        assert 'mutagroup_test_first_pack' not in sys.modules
        assert 'mutagroup_test_second_pack' not in sys.modules

        changer = MutationsSuperCollection.get_mutations().get_changer('answer = 0')
        applied = sorted(
            (coordinate.converter_id, changer.apply_coordinate(coordinate))
            for coordinate in changer.iterate_coordinates()
        )
        assert applied == [('I-13', 'answer = 42'), ('J-7', 'answer = 99')], applied
    """)


def test_get_mutations_returns_empty_supercollection_without_plugins(tmp_path: Path) -> None:
    """Build a usable supercollection when the slot has no plugins."""
    _run_with_plugins(tmp_path, """
        changer = MutationsSuperCollection.get_mutations().get_changer('answer = 0')
        assert list(changer.iterate_coordinates()) == []
    """)


def test_get_mutations_uses_custom_metacode_key(two_pack_plugins: Path) -> None:
    """Use the requested comment key for the discovered mutation packs."""
    _run_with_plugins(two_pack_plugins, """
        source = 'answer = 0  # custom: ignore[I-13]'
        default = MutationsSuperCollection.get_mutations().get_changer(source)
        custom = MutationsSuperCollection.get_mutations(metacode_key='custom').get_changer(source)

        assert sorted(item.converter_id for item in default.iterate_coordinates()) == ['I-13', 'J-7']
        assert [item.converter_id for item in custom.iterate_coordinates()] == ['J-7']
    """)


def test_get_mutations_rejects_duplicate_prefixes(tmp_path: Path) -> None:
    """Reject two plugins that expose packs with the same prefix."""
    _write_pack_plugin(tmp_path, 'mutagroup_test_first_pack', 'I', 13, '42')
    _write_pack_plugin(tmp_path, 'mutagroup_test_second_pack', 'I', 7, '99')
    _run_with_plugins(tmp_path, """
        import pytest

        with pytest.raises(ValueError, match="prefix 'I'"):
            MutationsSuperCollection.get_mutations()
    """)
