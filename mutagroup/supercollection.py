from collections import defaultdict
from dataclasses import replace
from functools import cached_property
from keyword import iskeyword
from typing import Dict, Generator, Iterable, List, NamedTuple, Optional, Type

from cstvis import Changer, Collector, Converter, Coordinate
from cstvis.converter import resolve_converters
from cstvis.source_offsets import SourceOffsetResolver
from cstvis.visitors.bloodhound import Bloodhound
from cstvis.visitors.comments_aggregator import CommentsAggregator
from cstvis.wrapper import CallableWrapper
from libcst import Comment, CSTNode
from libcst.metadata import PositionProvider

from mutagroup.collection import MutationsCollection
from mutagroup.ignores import DEFAULT_METACODE_KEY, CommentIgnores, ignored_ids
from mutagroup.slot import mutations

_FilterIndex = Dict[Type[CSTNode], List[CallableWrapper[bool]]]


def _matching_filters(index: _FilterIndex, node_type: Type[CSTNode]) -> List[CallableWrapper[bool]]:
    matches = index.get(node_type, []) + index.get(CSTNode, [])  # type: ignore[type-abstract]
    return list(dict.fromkeys(matches, True))


class _PackFilters(NamedTuple):
    prefix: str
    by_type: _FilterIndex


class _Candidate(NamedTuple):
    converter: Converter
    mutation_id: str
    prefix: Optional[str]
    pack_filters: List[CallableWrapper[bool]]


class _ScopedDiscovery(Bloodhound):
    def __init__(
        self,
        converters: Dict[Type[CSTNode], List[Converter]],
        comments: Dict[int, str],
        source_offsets: SourceOffsetResolver,
        pack_by_converter: Dict[Converter, _PackFilters],
        filters: _FilterIndex,
    ) -> None:
        super().__init__(converters, comments, filters, source_offsets)
        self._pack_by_converter = pack_by_converter
        self._candidates_by_type: Dict[Type[CSTNode], List[_Candidate]] = {}

    def _candidates(self, node_type: Type[CSTNode]) -> List[_Candidate]:
        candidates = self._candidates_by_type.get(node_type)
        if candidates is None:
            candidates = []
            for converter, mutation_id in resolve_converters(self.nodes_mapping, node_type).items():
                pack = self._pack_by_converter.get(converter)
                if pack is None:
                    candidates.append(_Candidate(converter, mutation_id, None, []))
                else:
                    candidates.append(_Candidate(converter, mutation_id, pack.prefix, _matching_filters(pack.by_type, node_type)))
            self._candidates_by_type[node_type] = candidates
        return candidates

    def on_visit(self, node: CSTNode) -> bool:
        candidates = self._candidates(type(node))
        if not candidates:
            return True

        position = self.get_metadata(PositionProvider, node)
        coordinate = Coordinate(
            file=None,
            class_name=type(node).__name__,
            start_line=position.start.line,
            start_column=position.start.column,
            end_line=position.end.line,
            end_column=position.end.column,
        )
        context = self._build_context(node, coordinate)
        if self.filters and not all(predicate.call_with_context(node, context) for predicate in _matching_filters(self.filters, type(node))):
            return True

        accepted_packs: Dict[Optional[str], bool] = {}
        for candidate in candidates:
            if candidate.prefix not in accepted_packs:
                accepted_packs[candidate.prefix] = all(predicate.call_with_context(node, context) for predicate in candidate.pack_filters)
            if accepted_packs[candidate.prefix] and all(predicate.call_with_context(node, context) for predicate in candidate.converter.filters):
                self.coordinates.append(replace(coordinate, converter_id=candidate.mutation_id))
        return True


class _ScopedChanger(Changer):
    def __init__(self, source: str, collector: Collector, pack_by_converter: Dict[Converter, _PackFilters]) -> None:
        super().__init__(source, collector=collector)
        self._pack_by_converter = pack_by_converter

    @cached_property
    def _comments_by_lines(self) -> Dict[int, str]:
        wrapper, _ = self._traversal
        aggregator = CommentsAggregator()
        # The position index already contains the comment nodes in this module.
        with aggregator.resolve(wrapper):
            for node in wrapper.resolve(PositionProvider):
                if isinstance(node, Comment):
                    aggregator.on_visit(node)
        return aggregator.comments

    def iterate_coordinates(self) -> Generator[Coordinate, None, None]:
        wrapper, source_offsets = self._traversal
        discovery = _ScopedDiscovery(
            self.converters_by_types,
            self._comments_by_lines,
            source_offsets,
            self._pack_by_converter,
            self.filters_by_types,
        )
        wrapper.visit(discovery)
        yield from discovery.coordinates


class MutationsSuperCollection:
    """Prepare one cstvis Changer per source for a set of mutation packs."""

    def __init__(self, collections: Iterable[MutationsCollection], *, metacode_key: str = DEFAULT_METACODE_KEY) -> None:
        if not metacode_key.isidentifier() or iskeyword(metacode_key):
            raise ValueError('The metacode key must be a valid Python identifier.')

        by_prefix: Dict[str, MutationsCollection] = {}
        for collection in collections:
            if collection.prefix in by_prefix:
                raise ValueError(f'Mutation collection prefix {collection.prefix!r} is already present.')
            by_prefix[collection.prefix] = collection
        self._collections = tuple(by_prefix.values())
        self._metacode_key = metacode_key

    @classmethod
    def get_mutations(cls, metacode_key: str = 'muta') -> 'MutationsSuperCollection':
        return cls(list(mutations().values()), metacode_key=metacode_key)

    def get_changer(self, source: str, *, ignored: Iterable[str] = ()) -> Changer:
        """Snapshot the packs and discover their coordinates together in this source."""
        project_ignores = ignored_ids(ignored)
        comment_ignores = CommentIgnores(self._metacode_key)
        collector = Collector()
        pack_by_converter: Dict[Converter, _PackFilters] = {}
        for collection in self._collections:
            pack = collection._snapshot(project_ignores, comment_ignores)
            collector._converters.extend(pack._converters)
            filters: _FilterIndex = defaultdict(list)
            for predicate in pack._filters:
                for node_type in predicate.first_node_annotations:
                    filters[node_type].append(predicate)
            scope = _PackFilters(collection.prefix, filters)
            pack_by_converter.update(dict.fromkeys(pack._converters, scope))
        return _ScopedChanger(source, collector, pack_by_converter)
