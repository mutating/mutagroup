from collections import Counter
from importlib.metadata import version
from typing import List, Optional, Tuple, Union

import pytest
from cstvis import Changer, Coordinate
from libcst import Add, BinaryOperation, Float, Subtract
from metacode import ParsedComment

from mutagroup import Context, MutationsCollection, MutationsSuperCollection


@pytest.mark.parametrize(
    ('directive', 'expected'),
    [
        pytest.param('muta: ignore[I-13]', ['I-1'], id='bare-id'),
        pytest.param('muta: ignore["I-13"]', ['I-1'], id='quoted-id'),
        pytest.param('muta: ignore[I]', [], id='bare-prefix'),
        pytest.param('muta: ignore["I"]', [], id='quoted-prefix'),
        pytest.param('muta: ignore[I-130, I-1]', ['I-13'], id='exact-id-match'),
        pytest.param('muta: ignore[II]', ['I-13', 'I-1'], id='exact-prefix-match'),
        pytest.param('muta: ignore[i-13]', ['I-13', 'I-1'], id='id-case'),
        pytest.param('MUTA: ignore[I]', ['I-13', 'I-1'], id='key-case'),
        pytest.param('muta: ignore[I - 13, "I-1"]', [], id='mixed-id-list'),
        pytest.param('muta: ignore[I-13] # muta: ignore[I-1]', [], id='chained-directives'),
        pytest.param('muta: ignore', ['I-13', 'I-1'], id='missing-arguments'),
        pytest.param('muta: ignore[]', ['I-13', 'I-1'], id='empty-arguments'),
        pytest.param('muta: ignore_next_line[I-13]', ['I-13', 'I-1'], id='similar-command-name'),
        pytest.param('muta: note["I-1"] # muta: ignore["I-13"]', ['I-1'], id='ignore-after-unrelated-command'),
        pytest.param('muta: ignore[I+13, I-True, I-13.0, I-f(), (I+J)-13, None]', ['I-13', 'I-1'], id='unsupported-expressions'),
        pytest.param('muta: ignore[I+13, "I-1"]', ['I-13'], id='valid-id-with-unsupported-expression'),
        pytest.param('muta: note["# muta ignore I-13 "]', ['I-13', 'I-1'], id='quoted-hash'),
        pytest.param(
            'muta: note["# muta: ignore[I-13] #"]',
            ['I-13', 'I-1'],
            id='quoted-canonical-directive',
            marks=pytest.mark.xfail(
                version('metacode') == '0.0.7',
                reason='metacode 0.0.7 reparses the text after a hash inside a string argument',
                raises=AssertionError,
                strict=True,
            ),
        ),
        pytest.param('other: note["# muta ignore I-13 #"]', ['I-13', 'I-1'], id='quoted-hash-under-other-key'),
        pytest.param('other: ignore[I]', ['I-13', 'I-1'], id='unrelated-key'),
        pytest.param('type: typing.List[int]', ['I-13', 'I-1'], id='qualified-type-comment'),
        pytest.param('type: typing.List[int] # muta: ignore[I-13]', ['I-1'], id='qualified-type-before-ignore'),
        pytest.param('muta: ignore[I-13] # type: typing.List[int]', ['I-1'], id='qualified-type-after-ignore'),
        pytest.param('muta: ignore[I-13] # type: typing.List[int] # muta: ignore[I-1]', [], id='chained-directives-around-qualified-type'),
        pytest.param(
            'type: typing.List["# muta: ignore[I-13] #"] # muta: ignore["I-1"]',
            ['I-13'],
            id='quoted-hash-in-qualified-type',
        ),
        pytest.param('muta: ignore[bad(] # muta: ignore[I-13]', ['I-1'], id='invalid-segment-before-ignore'),
        pytest.param('muta here', ['I-13', 'I-1'], id='ordinary-prose'),
        pytest.param('muta ignore I-13', ['I-1'], id='legacy-id'),
        pytest.param('explanation # muta ignore I-13, X-1 I-1', [], id='legacy-list'),
        pytest.param('muta ignore I-130', ['I-13', 'I-1'], id='legacy-exact-id-match'),
        pytest.param('muta ignore', ['I-13', 'I-1'], id='legacy-missing-arguments'),
        pytest.param('note muta ignore I-13', ['I-13', 'I-1'], id='legacy-in-prose'),
        pytest.param('muta ignore I-13 # unfinished [', ['I-1'], id='legacy-with-incomplete-text'),
    ],
)
def test_inline_mutes_match_only_explicit_directives(directive: str, expected: List[str]) -> None:
    """Mute exact IDs and pack prefixes while leaving unrelated comment content alone."""
    pack = MutationsCollection('I')

    def keep(node: Add) -> Add:
        return node

    pack.mutation(13)(keep)
    pack.mutation(keep)
    changer = Changer(f'1 + 2 # {directive}', collector=pack.collector())
    assert Counter(coordinate.converter_id for coordinate in changer.iterate_coordinates()) == Counter(expected)


def test_reused_collector_keeps_inline_mutes_local_to_each_source() -> None:
    """Reuse a collector on two sources with different comments at the same line number."""
    pack = MutationsCollection('I')

    @pack.mutation(13)
    def subtract(node: Add) -> Subtract:
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    collector = pack.collector()
    muted = Changer('1 + 2  # muta: ignore[I-13]', collector=collector)
    active = Changer('3 + 4  # keep', collector=collector)

    assert list(muted.iterate_coordinates()) == []
    coordinates = list(active.iterate_coordinates())
    assert coordinates == [Coordinate(None, 'Add', 1, 2, 1, 3, 'I-13')]
    assert active.apply_coordinate(coordinates[0]) == '3 - 4  # keep'
    assert list(muted.iterate_coordinates()) == []


@pytest.mark.parametrize(
    ('identifier', 'expected_ids'),
    [
        pytest.param('other-pack', ['other-pack-13-13', 'other-packer-13'], id='pack-prefix'),
        pytest.param('other-pack-13', ['other-pack-1', 'other-packer-13'], id='shared-id-and-prefix'),
        pytest.param('"other-pack-13"', ['other-pack-1', 'other-packer-13'], id='quoted-id-and-prefix'),
    ],
)
def test_hyphenated_prefixes_in_metacode(identifier: str, expected_ids: List[str]) -> None:
    """Match a hyphenated prefix or ID exactly for every type in the mutation's union."""
    pack = MutationsCollection('other-pack')
    matching_prefix_pack = MutationsCollection('other-pack-13')
    neighboring_pack = MutationsCollection('other-packer')

    def keep(node: Union[Add, Float]) -> Union[Add, Float]:
        return node

    pack.mutation(13)(keep)
    pack.mutation(keep)
    matching_prefix_pack.mutation(13)(keep)
    neighboring_pack.mutation(13)(keep)
    changer = MutationsSuperCollection([pack, matching_prefix_pack, neighboring_pack]).get_changer(f'1 + 2.5 # muta: ignore[{identifier}]')
    assert Counter(
        (coordinate.class_name, coordinate.converter_id) for coordinate in changer.iterate_coordinates()
    ) == Counter((node_type, mutation_id) for mutation_id in expected_ids for node_type in ('Add', 'Float'))


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize('comment', ['muta: ignore[I-13]', 'muta ignore I-13'], ids=['canonical', 'legacy'])
@pytest.mark.parametrize('placement', ['preceding-line', 'string-literal', 'string-before-comment'])
def test_mute_directives_in_strings_or_on_preceding_lines_are_ignored(combined: bool, comment: str, placement: str) -> None:
    """Keep candidates when directive-shaped text is on another line or inside a Python string.

    Read the trailing comment through Context to check its boundary beside a string.
    """
    pack = MutationsCollection('I')
    seen_comments: List[Optional[str]] = []

    @pack.mutation(13)
    def keep(node: Add) -> Add:
        return node

    @pack.filter
    def record_comment(_node: Add, context: Context) -> bool:
        seen_comments.append(context.comment)
        return True

    source = {
        'preceding-line': f'# {comment}\n1 + 2',
        'string-literal': f'value = "input" + """# {comment}\n"""',
        'string-before-comment': f'value = "# {comment} #" + "tail"  # marker',
    }[placement]
    changer = MutationsSuperCollection([pack]).get_changer(source) if combined else Changer(source, collector=pack.collector())
    (coordinate,) = changer.iterate_coordinates()
    assert coordinate.converter_id == 'I-13'
    expected_comment = ' marker' if placement == 'string-before-comment' else None
    assert seen_comments == [expected_comment]


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize(
    ('comment_line', 'expected_spans'),
    [
        pytest.param(1, [(2, 4, 3, 5), (5, 7, 5, 12)], id='opening-parenthesis'),
        pytest.param(2, [(5, 7, 5, 12)], id='node-start'),
        pytest.param(3, [(2, 4, 3, 5), (5, 7, 5, 12)], id='node-end'),
        pytest.param(4, [(2, 4, 3, 5), (5, 7, 5, 12)], id='closing-parenthesis'),
    ],
)
def test_multiline_node_uses_its_start_line_for_comments_and_mutes(
    combined: bool,
    comment_line: int,
    expected_spans: List[Tuple[int, int, int, int]],
) -> None:
    """Use the node's start line for both muting and the mutation handler's comment.

    Apply each surviving coordinate to check its comment and the source surrounding the parenthesized expression.
    """
    lines = ['v = (', '    1 +', '    2', ')', 'tail = 3 + 4']
    if comment_line != 2:
        lines[1] += '  # node-start'
    lines[comment_line - 1] += '  # muta: ignore[I-13]'
    source = '\n'.join(lines) + '\n'

    pack = MutationsCollection('I')
    received_comments: List[Optional[str]] = []

    @pack.mutation(13)
    def keep(node: BinaryOperation, context: Context) -> BinaryOperation:
        received_comments.append(context.comment)
        position = context.position
        if position.coordinate.start_line == 2:
            assert position.code_before == 'v = '
            assert position.code_after == f'{lines[3][1:]}\n{lines[4]}\n'
        return node

    changer = MutationsSuperCollection([pack]).get_changer(source) if combined else Changer(source, collector=pack.collector())
    coordinates = list(changer.iterate_coordinates())
    assert coordinates == [
        Coordinate(None, 'BinaryOperation', *span, 'I-13') for span in expected_spans
    ]
    assert [changer.apply_coordinate(coordinate) for coordinate in coordinates] == [source] * len(expected_spans)
    expected_comments: List[Optional[str]] = [None] if comment_line == 2 else [' node-start', None]
    assert received_comments == expected_comments


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined-custom-key'])
def test_default_mute_filter_uses_directives_from_the_cstvis_context(combined: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the directive returned by Context control suppression for a plain source comment.

    A direct parse of the plain comment returns an empty result.
    """
    pack = MutationsCollection('I')

    def keep(node: Add) -> Add:
        return node

    pack.mutation(13)(keep)
    pack.mutation(keep)
    expected_key = 'engine' if combined else 'muta'

    def read(context: Context, key: str) -> List[ParsedComment]:
        assert context.comment == ' marker'
        assert key == expected_key
        return [ParsedComment(key, 'ignore', ['I-13'])]

    monkeypatch.setattr(Context, 'get_metacodes', read)

    source = '1 + 2  # marker'
    if combined:
        changer = MutationsSuperCollection([pack], metacode_key=expected_key).get_changer(source)
    else:
        changer = Changer(source, collector=pack.collector())
    assert [coordinate.converter_id for coordinate in changer.iterate_coordinates()] == ['I-1']
