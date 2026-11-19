import sys
from copy import copy
from functools import update_wrapper
from re import fullmatch
from threading import Lock
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from cstvis import Collector, Context, Converter
from cstvis.converter import ConverterFilter, ConverterFunction
from cstvis.wrapper import CallableWrapper, _unwrap_partial
from libcst import CSTNode

from mutagroup.ignores import (
    DEFAULT_METACODE_KEY,
    CommentIgnores,
    ignored_ids,
    make_ignore_filter,
)

if sys.version_info >= (3, 10):  # pragma: no cover (<py310)
    from typing import ParamSpec
else:  # pragma: no cover (>=py310)
    from typing_extensions import ParamSpec

_Parameters = ParamSpec('_Parameters')
_Replacement = TypeVar('_Replacement', bound=CSTNode)
_FilterParameters = ParamSpec('_FilterParameters')
_Filter = Callable[_FilterParameters, bool]
_Meta = Optional[Dict[str, object]]


class _NodeFilter(CallableWrapper[bool]):
    def call_with_context(self, node: CSTNode, context: Context) -> bool:
        node_types = self.first_node_annotations
        if type(node) in node_types or CSTNode in node_types:
            return super().call_with_context(node, context)
        return True


class _MutationConverter(Converter):
    def __init__(self, mutation_id: str, function: ConverterFunction, meta: _Meta) -> None:
        self.mutation_id = mutation_id
        handler = cast(object, _unwrap_partial(function))
        self.mutation_name = cast(str, getattr(handler, '__name__', type(handler).__name__))
        super().__init__(function, meta=self._with_identity(meta))

    def get_function_id(self) -> str:
        return self.mutation_id

    def _with_identity(self, meta: _Meta) -> Dict[str, object]:
        return {**(meta or {}), 'mutation_id': self.mutation_id, 'mutation_name': self.mutation_name}


class Mutation(Generic[_Parameters, _Replacement]):
    """A registered mutation that can be called directly or given bound filters."""

    def __init__(
        self,
        converter: _MutationConverter,
        function: Callable[_Parameters, _Replacement],
    ) -> None:
        self._converter = converter
        self._function = function
        update_wrapper(self, function, updated=())

    @property
    def id(self) -> str:
        return self._converter.mutation_id

    def __call__(self, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Replacement:
        return self._function(*args, **kwargs)

    @overload  # pragma: no cover
    def filter(self, function: _Filter[_FilterParameters], meta: _Meta = None) -> _Filter[_FilterParameters]: ...

    @overload  # pragma: no cover
    def filter(
        self, function: None = None, meta: _Meta = None,
    ) -> Callable[[_Filter[_FilterParameters]], _Filter[_FilterParameters]]: ...

    def filter(
        self,
        function: Optional[_Filter[_FilterParameters]] = None,
        meta: _Meta = None,
    ) -> Union[_Filter[_FilterParameters], Callable[[_Filter[_FilterParameters]], _Filter[_FilterParameters]]]:
        """Register a filter for this mutation on the node types it annotates."""

        def register(predicate: _Filter[_FilterParameters]) -> _Filter[_FilterParameters]:
            self._converter.filters.append(_NodeFilter(cast(ConverterFilter, predicate), meta=self._converter._with_identity(meta)))
            return predicate

        return register if function is None else register(function)


class MutationsCollection:
    """Register mutations and node-type filters for one pack of mutations."""

    def __init__(self, prefix: str, *, meta: _Meta = None) -> None:
        if not isinstance(prefix, str) or not fullmatch(r'[^\s,#]+', prefix):
            raise ValueError('The mutation collection prefix must be a nonempty token without whitespace, commas or #.')

        self._prefix = prefix
        self._collector = Collector(meta=meta)
        self._converters: Dict[int, _MutationConverter] = {}
        self._next_number = 1
        self._lock = Lock()

    @property
    def prefix(self) -> str:
        return self._prefix

    @overload  # pragma: no cover
    def mutation(
        self, number: Callable[_Parameters, _Replacement], *, meta: _Meta = None,
    ) -> Mutation[_Parameters, _Replacement]: ...

    @overload  # pragma: no cover
    def mutation(
        self, number: Optional[int] = None, *, meta: _Meta = None,
    ) -> Callable[[Callable[_Parameters, _Replacement]], Mutation[_Parameters, _Replacement]]: ...

    def mutation(
        self,
        number: Union[int, Callable[_Parameters, _Replacement], None] = None,
        *,
        meta: _Meta = None,
    ) -> Union[
        Mutation[_Parameters, _Replacement],
        Callable[[Callable[_Parameters, _Replacement]], Mutation[_Parameters, _Replacement]],
    ]:
        """Register a mutation with an explicit number or the next free number."""
        if callable(number):
            return self._register_mutation(number, None, meta)
        if number is not None and (type(number) is not int or number < 1):
            raise ValueError('Mutation numbers must be positive integers.')

        def register(
            function: Callable[_Parameters, _Replacement],
        ) -> Mutation[_Parameters, _Replacement]:
            return self._register_mutation(function, number, meta)

        return register

    def _register_mutation(
        self,
        function: Callable[_Parameters, _Replacement],
        number: Optional[int],
        meta: _Meta,
    ) -> Mutation[_Parameters, _Replacement]:
        with self._lock:
            if number is None:
                while self._next_number in self._converters:
                    self._next_number += 1
                number = self._next_number
            elif number in self._converters:
                raise ValueError(f'Mutation {self.prefix}-{number} is already registered.')

            converter = _MutationConverter(
                f'{self.prefix}-{number}',
                cast(ConverterFunction, function),
                meta=cast(_Meta, self._collector._merge_meta(meta)),
            )
            mutation = Mutation(converter, function)
            self._converters[number] = converter
            return mutation

    @overload  # pragma: no cover
    def filter(self, function: _Filter[_FilterParameters], meta: _Meta = None) -> _Filter[_FilterParameters]: ...

    @overload  # pragma: no cover
    def filter(
        self, function: None = None, meta: _Meta = None,
    ) -> Callable[[_Filter[_FilterParameters]], _Filter[_FilterParameters]]: ...

    def filter(
        self,
        function: Optional[_Filter[_FilterParameters]] = None,
        meta: _Meta = None,
    ) -> Union[_Filter[_FilterParameters], Callable[[_Filter[_FilterParameters]], _Filter[_FilterParameters]]]:
        """Register a cstvis node-type filter for this collection."""

        def register(predicate: _Filter[_FilterParameters]) -> _Filter[_FilterParameters]:
            with self._lock:
                self._collector.filter(cast(ConverterFilter, predicate), meta=meta)
            return predicate

        return register if function is None else register(function)

    def collector(self, *, ignored: Iterable[str] = ()) -> Collector:
        """Build a snapshot of this pack for a cstvis Changer."""
        return self._snapshot(ignored_ids(ignored), CommentIgnores(DEFAULT_METACODE_KEY))

    def _snapshot(self, project_ignores: FrozenSet[str], comment_ignores: CommentIgnores) -> Collector:
        with self._lock:
            collector = self._collector + Collector()
            for mutation in self._converters.values():
                converter = copy(mutation)
                default_filter = make_ignore_filter(mutation.mutation_id, self.prefix, project_ignores, comment_ignores)
                converter.filters = [default_filter, *mutation.filters]
                # Collector.converter creates its own Converter with a function-based ID.
                collector._converters.append(converter)
        return collector
