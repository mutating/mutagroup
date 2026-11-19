from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from inspect import Signature, signature
from operator import or_
from sys import version_info
from threading import Barrier
from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

import pytest
from cstvis import Changer, Collector, Context as CSTContext, Converter, Coordinate
from libcst import (
    Add,
    BaseExpression,
    CSTNode,
    Float,
    Integer,
    Multiply,
    SimpleString,
    SimpleWhitespace,
    Subtract,
)
from metacode import ParsedComment
from sigmatch.errors import SignatureMismatchError

from mutagroup import Context, Mutation, MutationsCollection, MutationsSuperCollection


def test_mutations_support_explicit_and_automatic_numbers() -> None:
    """Keep each registration's ID, handler name, and metadata with either numbering mode."""
    mutations = MutationsCollection('I')
    register_later = mutations.mutation()
    seen_ids: List[object] = []

    def subtract(node: Add, context: Context) -> Subtract:
        assert context.meta is not None
        assert context.meta['mutation_name'] == 'subtract'
        mutation_id = context.meta['mutation_id']
        seen_ids.append(mutation_id)
        expected_label = 'explicit' if mutation_id == 'I-13' else None
        assert context.meta.get('label') == expected_label
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    explicit_thirteen = mutations.mutation(13, meta={'label': 'explicit'})(subtract)
    explicit_two = mutations.mutation(2)(subtract)

    def multiply(node: Add, context: Context) -> Multiply:
        assert context.meta is not None
        assert context.meta['mutation_name'] == 'multiply'
        assert context.meta['label'] == 'automatic'
        seen_ids.append(context.meta['mutation_id'])
        return Multiply(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    automatic_one = mutations.mutation(multiply, meta={'label': 'automatic'})
    automatic_three = register_later(subtract)

    assert mutations.prefix == 'I'
    assert [explicit_thirteen.id, explicit_two.id, automatic_one.id, automatic_three.id] == ['I-13', 'I-2', 'I-1', 'I-3']

    changer = Changer('1 + 2', collector=mutations.collector())
    coordinates = list(changer.iterate_coordinates())
    assert Counter(
        (coordinate.converter_id, changer.apply_coordinate(coordinate)) for coordinate in coordinates
    ) == Counter([('I-13', '1 - 2'), ('I-2', '1 - 2'), ('I-1', '1 * 2'), ('I-3', '1 - 2')])
    assert seen_ids == [coordinate.converter_id for coordinate in coordinates]


def test_mutation_handle_keeps_signature_and_can_be_called_directly() -> None:
    """Preserve the callback signature and defaults, and let direct calls bypass discovery filters."""
    pack = MutationsCollection('I')
    filtered_values: List[str] = []

    def replace(node: Integer, *, value: str = '0') -> Integer:
        return node.with_changes(value=value)

    def reject(node: Integer) -> bool:
        filtered_values.append(node.value)
        return False

    mutation = pack.mutation()(replace)
    assert isinstance(mutation, Mutation)
    assert mutation.id == 'I-1'
    assert signature(mutation) == signature(replace)
    changer = Changer('1', collector=pack.collector())
    (coordinate,) = changer.iterate_coordinates()
    assert changer.apply_coordinate(coordinate) == '0'

    assert mutation.filter(reject) is reject
    assert mutation(Integer('1')).value == '0'
    assert mutation(Integer('1'), value='7').value == '7'
    assert filtered_values == []
    assert list(Changer('1', collector=pack.collector()).iterate_coordinates()) == []
    assert filtered_values == ['1']


@pytest.mark.parametrize('prefix', ['', 'two words', 'comma,', '#comment', 5])
def test_invalid_prefix_is_rejected(prefix: str) -> None:
    """Reject collection prefixes that cannot unambiguously identify a pack in a directive."""
    with pytest.raises(ValueError, match='prefix'):
        MutationsCollection(prefix)


@pytest.mark.parametrize('number', [0, -1, True, '13'])
def test_invalid_number_is_rejected(number: int) -> None:
    """Require positive integer mutation numbers so explicit IDs remain unambiguous."""
    mutations = MutationsCollection('I')
    with pytest.raises(ValueError, match='positive integers'):
        mutations.mutation(number)


@pytest.mark.parametrize('explicit_first', [True, False], ids=['explicit-id', 'automatic-id'])
@pytest.mark.parametrize('prepare_early', [True, False], ids=['early-decorator', 'late-decorator'])
def test_duplicate_number_is_rejected_without_replacing_its_handler(explicit_first: bool, prepare_early: bool) -> None:
    """Protect the first handler even if its competitor's decorator was prepared earlier."""
    mutations = MutationsCollection('I')

    def set_two(node: Integer) -> Integer:
        return node.with_changes(value='2')

    def set_three(node: Integer) -> Integer:
        return node.with_changes(value='3')

    retry = mutations.mutation(1) if prepare_early else None
    first = mutations.mutation(1)(set_two) if explicit_first else mutations.mutation(set_two)
    with pytest.raises(ValueError, match='I-1'):
        (retry if retry is not None else mutations.mutation(1))(set_three)

    assert first.id == 'I-1'
    assert mutations.mutation(set_three).id == 'I-2'
    changer = Changer('1', collector=mutations.collector())
    assert Counter(
        (coordinate.converter_id, changer.apply_coordinate(coordinate)) for coordinate in changer.iterate_coordinates()
    ) == Counter([('I-1', '2'), ('I-2', '3')])


@pytest.mark.parametrize('role', ['mutation', 'pack-filter', 'bound-filter'])
@pytest.mark.parametrize('signature_case', ['no-node', 'extra-required-argument', 'partial-with-bound-node'])
def test_invalid_callback_signature_leaves_the_pack_usable(role: str, signature_case: str) -> None:
    """Reject callbacks cstvis cannot call while preserving ID allocation and existing filters."""
    mutations = MutationsCollection('I')

    def replace(node: Integer) -> Integer:
        return node.with_changes(value='7')

    existing = mutations.mutation(13)(replace)

    @mutations.filter
    def skip_two(node: Integer) -> bool:
        return node.value != '2'

    @existing.filter
    def skip_three(node: Integer) -> bool:
        return node.value != '3'

    def no_node() -> NoReturn:
        pytest.fail('A callback without a node parameter ran.')

    def extra_required(_node: Integer, _context: Context, _extra: object) -> NoReturn:
        pytest.fail('A callback with a third required parameter ran.')

    def receive_node_and_context(_node: Integer, _context: Context) -> NoReturn:
        pytest.fail('A callback with its node already bound ran.')

    callback: Callable[..., NoReturn]
    if signature_case == 'no-node':
        callback = no_node
    elif signature_case == 'extra-required-argument':
        callback = extra_required
    else:
        callback = partial(receive_node_and_context, Integer('1'))

    if role == 'mutation':
        with pytest.raises(SignatureMismatchError):
            mutations.mutation(callback)
    elif role == 'pack-filter':
        with pytest.raises(SignatureMismatchError):
            mutations.filter(callback)
    else:
        with pytest.raises(SignatureMismatchError):
            existing.filter(callback)

    assert mutations.mutation(replace).id == 'I-1'
    source = '(1, 2, 3)'
    changer = Changer(source, collector=mutations.collector())
    assert Counter(
        (source[item.start_column:item.end_column], item.converter_id) for item in changer.iterate_coordinates()
    ) == Counter([('1', 'I-13'), ('1', 'I-1'), ('3', 'I-1')])


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize('role', ['mutation', 'pack-filter', 'bound-filter', 'native-filter'])
def test_callback_errors_reach_the_caller(combined: bool, role: str) -> None:
    """Propagate the original error from a filter during discovery or a mutation during application."""
    pack = MutationsCollection('I')
    error = RuntimeError('callback failed')

    def raise_error(_node: Integer) -> NoReturn:
        raise error

    if role == 'mutation':
        pack.mutation(raise_error)
    else:

        @pack.mutation
        def keep(node: Integer) -> Integer:
            return node

        if role == 'pack-filter':
            pack.filter(raise_error)
        elif role == 'bound-filter':
            keep.filter(raise_error)

    source = 'x = 1'
    changer = MutationsSuperCollection([pack]).get_changer(source) if combined else Changer(source, collector=pack.collector())
    if role == 'native-filter':
        changer.filter(raise_error)  # type: ignore[arg-type]
    if role == 'mutation':
        (coordinate,) = changer.iterate_coordinates()
        with pytest.raises(RuntimeError) as raised:
            changer.apply_coordinate(coordinate)
    else:
        with pytest.raises(RuntimeError) as raised:
            list(changer.iterate_coordinates())
    assert raised.value is error


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize('role', ['mutation', 'pack-filter', 'bound-filter'])
def test_unsupported_node_annotation_is_rejected(combined: bool, role: str) -> None:
    """Surface cstvis's type error when a callback targets a type outside its CST vocabulary."""
    mutations = MutationsCollection('I')

    @mutations.mutation
    def keep(node: Integer) -> Integer:
        return node

    if role == 'mutation':

        @mutations.mutation
        def invalid_mutation(_node: bool) -> Integer:
            return Integer('0')
    else:

        def invalid_filter(_node: bool) -> bool:
            return True

        register = mutations.filter if role == 'pack-filter' else keep.filter
        register(invalid_filter)

    source = '1'
    with pytest.raises(TypeError, match='type annotation'):
        list((
            MutationsSuperCollection([mutations]).get_changer(source)
            if combined else Changer(source, collector=mutations.collector())
        ).iterate_coordinates())


@pytest.mark.parametrize(
    ('number', 'expected_ids'),
    [
        pytest.param(None, ['I-1', 'I-2', 'I-3', 'I-4'], id='automatic'),
        pytest.param(13, ['I-13'], id='shared-explicit-number'),
    ],
)
def test_concurrent_registration_preserves_unique_ids(number: Optional[int], expected_ids: List[str]) -> None:
    """Assign each automatic ID once and accept only one claim to a shared explicit number.

    Start the threads together and yield during signature inspection to expose races in ID allocation.
    Give each thread a distinct result to verify that the accepted registrations keep their metadata.
    """
    pack = MutationsCollection('I')
    worker_count = 4
    ready = Barrier(worker_count, timeout=10)

    class SlowToInspect:
        @property
        def __signature__(self) -> Signature:
            sleep(0.01)
            return signature(self.__call__)

        def __call__(self, node: Integer, context: Context) -> Integer:
            assert context.meta is not None
            return node.with_changes(value=context.meta['value'])

    handler = SlowToInspect()

    def register(value: str) -> Tuple[str, str]:
        ready.wait()
        meta: Dict[str, object] = {'value': value}
        mutation = pack.mutation(handler, meta=meta) if number is None else pack.mutation(number, meta=meta)(handler)
        return mutation.id, value

    registered: List[Tuple[str, str]] = []
    rejections: List[str] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        registrations = [executor.submit(register, str(index + 2)) for index in range(worker_count)]
        for registration in registrations:
            try:
                registered.append(registration.result())
            except ValueError as error:
                rejections.append(str(error))

    assert sorted(mutation_id for mutation_id, _ in registered) == expected_ids
    assert len(rejections) == worker_count - len(expected_ids)
    assert all(f'I-{number}' in message for message in rejections)
    changer = Changer('1', collector=pack.collector())
    assert Counter(
        (coordinate.converter_id, changer.apply_coordinate(coordinate)) for coordinate in changer.iterate_coordinates()
    ) == Counter(registered)


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize('role', ['mutation', 'pack-filter', 'bound-filter'])
@pytest.mark.parametrize(
    ('annotation', 'matched'),
    [
        pytest.param(Add, ('Add',), id='concrete'),
        pytest.param(Union[Add, Integer], ('Integer', 'Add'), id='typing-union'),
        pytest.param(Union[int, float], ('Integer', 'Float'), id='union-of-aliases'),
        *([pytest.param(or_(Add, Integer), ('Integer', 'Add'), id='pep604-union')] if version_info >= (3, 10) else []),
        pytest.param(int, ('Integer',), id='int-alias'),
        pytest.param(float, ('Float',), id='float-alias'),
        pytest.param(str, ('SimpleString',), id='str-alias'),
        pytest.param(BaseExpression, (), id='non-wildcard-base-class'),
        pytest.param(CSTNode, None, id='cst-node'),
        pytest.param(Any, None, id='any'),
        pytest.param(None, None, id='unannotated'),
        pytest.param(Union[Add, CSTNode], None, id='overlapping-wildcard'),
    ],
)
def test_cstvis_dispatch_selects_the_expected_node_types(
    combined: bool,
    role: str,
    annotation: Optional[object],
    matched: Optional[Tuple[str, ...]],
) -> None:
    """Use cstvis dispatch for all callback roles, including exact types, aliases, and wildcards.

    Vary the runtime node annotation so each row uses the same source and callback behavior.
    The bare string expression shares its span with its enclosing nodes, so applying its
    coordinate also checks that cstvis selects the intended node class.
    """
    source = 'total = 1 + 2.5\n"x"\n'
    target_types = ('Integer', 'Add', 'Float', 'SimpleString')
    expected = target_types if matched is None else matched
    seen: List[str] = []
    mutations = MutationsCollection('I')

    if role == 'mutation':

        def observe(node: CSTNode) -> CSTNode:
            seen.append(type(node).__name__)
            return node

        if annotation is None:
            observe.__annotations__.pop('node')
        else:
            observe.__annotations__['node'] = annotation
        mutations.mutation(observe)
    else:

        @mutations.mutation
        def keep(node: Union[Integer, Add, Float, SimpleString]) -> CSTNode:
            return node

        def reject(node: CSTNode) -> bool:
            seen.append(type(node).__name__)
            return False

        if annotation is None:
            reject.__annotations__.pop('node')
        else:
            reject.__annotations__['node'] = annotation
        register = mutations.filter if role == 'pack-filter' else keep.filter
        assert register(reject) is reject

    changer = MutationsSuperCollection([mutations]).get_changer(source) if combined else Changer(source, collector=mutations.collector())
    coordinates = list(changer.iterate_coordinates())
    if role == 'mutation':
        assert seen == []
        selected = [coordinate for coordinate in coordinates if coordinate.class_name in target_types]
        assert [coordinate.class_name for coordinate in selected] == list(expected)
        if matched is not None:
            assert coordinates == selected
        else:
            native = Changer(source)
            native.converter(observe)
            assert coordinates == [replace(coordinate, converter_id='I-1') for coordinate in native.iterate_coordinates()]
        for coordinate in coordinates:
            assert changer.apply_coordinate(coordinate) == source
            assert seen == [coordinate.class_name]
            seen.clear()
    else:
        assert [coordinate.class_name for coordinate in coordinates] == [name for name in target_types if name not in expected]
        assert seen == list(expected)


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
def test_collection_and_bound_filters_compose_in_order(combined: bool) -> None:
    """Evaluate pack filters before bound filters and stop at the first rejection.

    Register the same callback at both scopes with different and repeated metadata to check that each registration is kept.
    """
    mutations = MutationsCollection('I')
    events: Dict[int, List[str]] = {number: [] for number in range(1, 9)}

    def replace(node: Integer) -> Integer:
        return node.with_changes(value='0')

    first = mutations.mutation(13)(replace)
    mutations.mutation(99)(replace)

    def stage_filter(node: Integer, context: Context) -> bool:
        assert context.meta is not None
        scope = context.meta['scope']
        stage = context.meta['stage']
        value = int(node.value)
        events[value].append(f'{scope}-{stage}')
        return value != stage + context.meta['offset']

    for stage in (1, 2, 3, 3):
        mutations.filter(stage_filter, meta={'scope': 'pack', 'stage': stage, 'offset': 0})
        first.filter(stage_filter, meta={'scope': 'bound', 'stage': stage, 'offset': 3})

    source = 'numbers = (1, 2, 3, 4, 5, 6, 7)\nlast = 8  # muta: ignore[I-13]\n'
    changer = MutationsSuperCollection([mutations]).get_changer(source) if combined else Changer(source, collector=mutations.collector())
    lines = source.splitlines()
    assert Counter(
        (lines[item.start_line - 1][item.start_column:item.end_column], item.converter_id)
        for item in changer.iterate_coordinates()
    ) == Counter([('4', 'I-99'), ('5', 'I-99'), ('6', 'I-99'), ('7', 'I-13'), ('7', 'I-99'), ('8', 'I-99')])
    all_pack_filters = ['pack-1', 'pack-2', 'pack-3', 'pack-3']
    all_bound_filters = ['bound-1', 'bound-2', 'bound-3', 'bound-3']
    assert events == {
        1: all_pack_filters[:1],
        2: all_pack_filters[:2],
        3: all_pack_filters[:3],
        4: all_pack_filters + all_bound_filters[:1],
        5: all_pack_filters + all_bound_filters[:2],
        6: all_pack_filters + all_bound_filters[:3],
        7: all_pack_filters + all_bound_filters,
        8: all_pack_filters,
    }


@pytest.mark.parametrize(
    ('combined', 'filter_scope'),
    [
        pytest.param(False, 'pack', id='standalone-pack'),
        pytest.param(False, 'bound', id='standalone-bound'),
        pytest.param(True, 'pack', id='combined-pack'),
        pytest.param(True, 'bound', id='combined-bound'),
        pytest.param(True, 'changer', id='combined-changer'),
    ],
)
def test_filter_order_matches_cstvis_for_concrete_and_wildcard_types(combined: bool, filter_scope: str) -> None:
    """Preserve callback order before rejection when both concrete and wildcard filters match.

    Bind the wildcard filter's column as a keyword while cstvis supplies the node and context.
    """
    pack = MutationsCollection('I')
    calls: List[str] = []

    def keep(node: Integer) -> Integer:
        return node

    mutation = pack.mutation(keep)

    def accept_node_at_column(_node: CSTNode, context: Context, *, column: int) -> bool:
        calls.append('allow-wildcard')
        return context.position.coordinate.start_column == column

    def reject_any_node(_node: CSTNode) -> bool:
        calls.append('reject-wildcard')
        return False

    def accept_integer(_node: Integer) -> bool:
        calls.append('concrete')
        return True

    source = 'x = 1'
    native = Changer(source)
    converter = native.converter(keep)  # type: ignore[arg-type]
    assert isinstance(converter, Converter)
    register_filter = pack.filter if filter_scope == 'pack' else mutation.filter
    register_native_filter = converter.filter if filter_scope == 'bound' else native.filter
    predicates = (partial(accept_node_at_column, column=4), reject_any_node, accept_integer)
    for predicate in predicates:
        if filter_scope != 'changer':
            register_filter(predicate)
        register_native_filter(predicate)  # type: ignore[arg-type]
    assert list(native.iterate_coordinates()) == []
    native_calls = calls.copy()
    calls.clear()

    changer = MutationsSuperCollection([pack]).get_changer(source) if combined else Changer(source, collector=pack.collector())
    if filter_scope == 'changer':
        for predicate in predicates:
            changer.filter(predicate)  # type: ignore[arg-type]
    assert list(changer.iterate_coordinates()) == []
    assert calls == native_calls


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize('with_overrides', [False, True], ids=['pack-defaults', 'explicit-overrides'])
def test_callbacks_receive_native_context_and_isolated_metadata(combined: bool, with_overrides: bool) -> None:
    """Give each callback native source positions and its own copy of inherited or overridden metadata.

    The BOM, non-ASCII name, and CRLF distinguish original character offsets from
    byte offsets and offsets in normalized source.
    A shared object verifies that copying the metadata preserves its values by reference.
    """
    marker = object()
    pack_meta: Dict[str, object] = {
        'pack': 'I',
        'label': 'default',
        'marker': marker,
        'mutation_id': 'pack-meta-id',
        'mutation_name': 'pack-meta-name',
    }
    mutation_meta: Dict[str, object] = {'label': '', 'mutation_id': 'wrong-id', 'mutation_name': 'wrong-name'}
    bound_meta: Dict[str, object] = {'label': 'bound filter', 'mutation_id': 'wrong-id', 'mutation_name': 'wrong-name', 'marker': marker}
    pack = MutationsCollection('I', meta=pack_meta)
    seen: List[Tuple[str, Context]] = []

    def subtract(node: Add, ctx: Optional[Context] = None, /) -> Subtract:
        assert ctx is not None
        assert ctx.meta is not None
        assert 'scratch' not in ctx.meta
        ctx.meta['scratch'] = 'mutation'
        seen.append(('mutation', ctx))
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    @pack.filter(meta={'label': 'collection filter'} if with_overrides else None)
    def group_filter(_node: Add, *, context: Optional[Context] = None) -> bool:
        assert context is not None
        assert context.meta is not None
        assert 'scratch' not in context.meta
        context.meta['scratch'] = 'collection'
        seen.append(('collection', context))
        return True

    mutation = pack.mutation(meta=mutation_meta)(subtract) if with_overrides else pack.mutation(subtract)

    @mutation.filter(meta=bound_meta)
    def bound_filter(_node: Add, *, context: Optional[Context] = None) -> bool:
        assert context is not None
        assert context.meta is not None
        assert 'scratch' not in context.meta
        context.meta['scratch'] = 'bound'
        seen.append(('bound', context))
        return True

    source = '\ufeffé = (1 + 2)  # hint: note["x"]\r\nb = 3 + 4  #'
    changer = MutationsSuperCollection([pack]).get_changer(source) if combined else Changer(source, collector=pack.collector())
    coordinates = list(changer.iterate_coordinates())
    assert coordinates == [
        Coordinate(None, 'Add', 1, 7, 1, 8, 'I-1'),
        Coordinate(None, 'Add', 2, 6, 2, 7, 'I-1'),
    ]
    filter_calls = [('collection', 1), ('bound', 1), ('collection', 2), ('bound', 2)]
    assert [(role, context.position.coordinate.start_line) for role, context in seen] == filter_calls
    assert [changer.apply_coordinate(coordinates[index]) for index in (1, 0, 0)] == [
        source.replace('3 + 4', '3 - 4'),
        source.replace('1 + 2', '1 - 2'),
        source.replace('1 + 2', '1 - 2'),
    ]
    assert [(role, context.position.coordinate.start_line) for role, context in seen] == [
        *filter_calls, ('mutation', 2), ('mutation', 1), ('mutation', 1),
    ]

    expected_meta = {
        'collection': {
            'pack': 'I',
            'label': 'collection filter' if with_overrides else 'default',
            'scratch': 'collection',
            'marker': marker,
            'mutation_id': 'pack-meta-id',
            'mutation_name': 'pack-meta-name',
        },
        'bound': {'label': 'bound filter', 'mutation_id': 'I-1', 'mutation_name': 'subtract', 'scratch': 'bound', 'marker': marker},
        'mutation': {
            'pack': 'I',
            'label': '' if with_overrides else 'default',
            'mutation_id': 'I-1',
            'mutation_name': 'subtract',
            'scratch': 'mutation',
            'marker': marker,
        },
    }
    expected_contexts = {
        1: (7, 10, ' hint: note["x"]', [ParsedComment('hint', 'note', ['x'])]),
        2: (38, 41, '', []),
    }
    for role, context in seen:
        assert type(context) is Context is CSTContext
        assert context.meta == expected_meta[role]
        position = context.position
        assert position.source == source
        assert position.coordinate == replace(coordinates[position.coordinate.start_line - 1], converter_id=None)
        start, end, comment, metacodes = expected_contexts[position.coordinate.start_line]
        assert (position.start_offset, position.end_offset) == (start, end)
        assert position.code_before == source[:start]
        assert position.code_after == source[end:]
        assert context.comment == comment
        assert context.get_metacodes('hint') == metacodes

    assert pack_meta == {
        'pack': 'I',
        'label': 'default',
        'marker': marker,
        'mutation_id': 'pack-meta-id',
        'mutation_name': 'pack-meta-name',
    }
    assert mutation_meta == {'label': '', 'mutation_id': 'wrong-id', 'mutation_name': 'wrong-name'}
    assert bound_meta == {'label': 'bound filter', 'mutation_id': 'wrong-id', 'mutation_name': 'wrong-name', 'marker': marker}


@pytest.mark.parametrize('callback_kind', ['partial', 'callable-object'])
def test_mutation_name_describes_the_underlying_callable(callback_kind: str) -> None:
    """Record the function name for a partial and the class name for a callable instance."""
    names: List[str] = []

    def replace(node: Integer, *, value: str, context: Optional[Context] = None) -> Integer:
        assert context is not None
        assert context.meta is not None
        names.append(context.meta['mutation_name'])
        return node.with_changes(value=value)

    class Replacement:
        def __call__(self, node: Integer, context: Context) -> Integer:
            assert context.meta is not None
            names.append(context.meta['mutation_name'])
            return node.with_changes(value='7')

    callback: Callable[..., Integer]
    if callback_kind == 'partial':
        callback = partial(replace, value='7')
        expected_name = 'replace'
    else:
        callback = Replacement()
        expected_name = 'Replacement'

    pack = MutationsCollection('I')
    pack.mutation(callback)
    changer = Changer('1', collector=pack.collector())
    (coordinate,) = changer.iterate_coordinates()
    assert changer.apply_coordinate(coordinate) == '7'
    assert names == [expected_name]


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize('form', ['single-id', 'list', 'iterator'])
def test_project_mutes_compose_without_mutating_the_pack(combined: bool, form: str) -> None:
    """Combine project and line mutes for one run and skip bound filters on muted mutations.

    Mutes use discovery filters, so a caller can still apply a known coordinate directly.
    """
    mutations = MutationsCollection('I')

    @mutations.mutation
    def subtract(node: Add) -> Subtract:
        return Subtract(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    @mutations.mutation(13)
    def multiply(node: Add) -> Multiply:
        return Multiply(whitespace_before=node.whitespace_before, whitespace_after=node.whitespace_after)

    @subtract.filter
    def require_single_trailing_space(node: Add) -> bool:
        return node.whitespace_after.deep_equals(SimpleWhitespace(' '))

    multiply_filter_lines: List[int] = []

    @multiply.filter
    def record_line(_node: Add, context: Context) -> bool:
        multiply_filter_lines.append(context.position.coordinate.start_line)
        return True

    ids = ['I-13', 'I-13', 'I-130', 'I', 'unknown']
    ignored: Iterable[str] = {'single-id': 'I-13', 'list': ids, 'iterator': iter(ids)}[form]
    source = '1 + 2 # muta: ignore[I-1]\n3 + 4\n5+6\n'
    packs = MutationsSuperCollection([mutations])
    with_project_mutes = (
        packs.get_changer(source, ignored=ignored)
        if combined else Changer(source, collector=mutations.collector(ignored=ignored))
    )
    assert [(item.start_line, item.converter_id) for item in with_project_mutes.iterate_coordinates()] == [(2, 'I-1')]
    assert multiply_filter_lines == []

    without_project_mutes = packs.get_changer(source) if combined else Changer(source, collector=mutations.collector())
    available_coordinates = list(without_project_mutes.iterate_coordinates())
    assert Counter((item.start_line, item.converter_id) for item in available_coordinates) == Counter([
        (1, 'I-13'), (2, 'I-1'), (2, 'I-13'), (3, 'I-13'),
    ])
    assert multiply_filter_lines == [1, 2, 3]
    (known_coordinate,) = (item for item in available_coordinates if (item.start_line, item.converter_id) == (2, 'I-13'))
    assert with_project_mutes.apply_coordinate(known_coordinate) == source.replace('3 + 4', '3 * 4')
    inline_coordinate = Coordinate(None, 'Add', 1, 2, 1, 3, 'I-1')
    assert with_project_mutes.apply_coordinate(inline_coordinate) == source.replace('1 + 2', '1 - 2')


@pytest.mark.parametrize('combined', [False, True], ids=['standalone', 'combined'])
@pytest.mark.parametrize(
    ('late_registration', 'expected_mutants'),
    [
        ('mutation', [('I-1', 'x = 2'), ('I-2', 'x = 3'), ('I-3', 'x = 4')]),
        ('pack-filter', []),
        ('bound-filter', [('I-2', 'x = 3')]),
    ],
)
def test_late_registrations_only_affect_fresh_snapshots(
    combined: bool,
    late_registration: str,
    expected_mutants: List[Tuple[str, str]],
) -> None:
    """Keep collector and changer snapshots isolated from later registrations.

    Add one registration per case so a pack filter cannot mask a change to a bound filter.
    """
    mutations = MutationsCollection('I')
    packs = MutationsSuperCollection([mutations])

    @mutations.mutation
    def first(node: Integer) -> Integer:
        return node.with_changes(value='2')

    @mutations.mutation
    def second(node: Integer) -> Integer:
        return node.with_changes(value='3')

    source = 'x = 1'
    snapshot = packs.get_changer(source) if combined else mutations.collector()

    if late_registration == 'mutation':

        @mutations.mutation
        def third(node: Integer) -> Integer:
            return node.with_changes(value='4')
    elif late_registration == 'pack-filter':

        @mutations.filter
        def reject_pack(_node: Integer) -> bool:
            return False
    else:

        @first.filter
        def reject_first(_node: Integer) -> bool:
            return False

    original_changer = Changer(source, collector=snapshot) if isinstance(snapshot, Collector) else snapshot
    updated_changer = packs.get_changer(source) if combined else Changer(source, collector=mutations.collector())
    assert Counter(
        (coordinate.converter_id, original_changer.apply_coordinate(coordinate))
        for coordinate in original_changer.iterate_coordinates()
    ) == Counter([('I-1', 'x = 2'), ('I-2', 'x = 3')])
    assert Counter(
        (coordinate.converter_id, updated_changer.apply_coordinate(coordinate))
        for coordinate in updated_changer.iterate_coordinates()
    ) == Counter(expected_mutants)
