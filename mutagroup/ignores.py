from ast import BinOp, Constant, Name, Sub
from io import StringIO
from tokenize import COMMENT, TokenError, generate_tokens
from typing import Dict, FrozenSet, Iterable, Iterator, Optional, cast

from cstvis import Context
from cstvis.wrapper import CallableWrapper
from libcst import CSTNode, ParserSyntaxError
from metacode import UnknownArgumentTypeError, parse

DEFAULT_METACODE_KEY = 'muta'
_NO_IGNORES: FrozenSet[str] = frozenset()


def _identifier(argument: object) -> Optional[str]:
    if isinstance(argument, str):
        return argument
    if isinstance(argument, Name):
        return argument.id
    if isinstance(argument, BinOp) and isinstance(argument.op, Sub):
        prefix = _identifier(argument.left)
        suffix = argument.right
        if prefix is not None and isinstance(suffix, Name):
            return f'{prefix}-{suffix.id}'
        if prefix is not None and isinstance(suffix, Constant):  # type: ignore[misc]
            number = cast(object, suffix.value)
            if type(number) is int:
                return f'{prefix}-{number}'
    return None


def _comment_parts(comment: str) -> Iterator[str]:
    while comment:
        tokens = generate_tokens(StringIO(comment).readline)
        try:
            boundary = next((token.start[1] for token in tokens if token.type == COMMENT), len(comment))
        except TokenError:
            boundary = len(comment)
        yield comment[:boundary]
        comment = comment[boundary + 1:]


def _legacy_ids(comment: str, key: str) -> Iterator[str]:
    if key not in comment or 'ignore' not in comment:
        return
    for part in _comment_parts(comment):
        words = part.split(maxsplit=2)
        if len(words) == 3 and words[:2] == [key, 'ignore']:
            yield from words[2].replace(',', ' ').split()


class CommentIgnores:
    """Reuse parsed ignore directives for every mutation on a comment's line."""

    def __init__(self, key: str) -> None:
        self._key = key
        self._parsed: Dict[str, FrozenSet[str]] = {}

    def for_context(self, context: Context) -> FrozenSet[str]:
        comment = context.comment
        if comment is None:
            return _NO_IGNORES
        if comment not in self._parsed:
            self._parsed[comment] = self._read(context, comment)
        return self._parsed[comment]

    def _read(self, context: Context, comment: str) -> FrozenSet[str]:
        try:
            try:
                directives = context.get_metacodes(self._key)
            except UnknownArgumentTypeError:
                # metacode 0.0.7 represents an unquoted ID such as I-13 as a subtraction AST.
                directives = parse(comment, self._key, allow_ast=True)
        except (AttributeError, ParserSyntaxError):
            # A qualified annotation such as type: typing.List[int] can break the parser.
            directives = []
            for part in _comment_parts(comment):
                try:
                    directives.extend(parse(part, self._key, allow_ast=True))
                except (AttributeError, ParserSyntaxError):
                    continue

        if not directives:
            return frozenset(_legacy_ids(comment, self._key))

        return frozenset(
            code
            for directive in directives if directive.command == 'ignore'
            for argument in cast(Iterable[object], directive.arguments)
            if (code := _identifier(argument)) is not None
        )


def ignored_ids(ignored: Iterable[str]) -> FrozenSet[str]:
    return frozenset((ignored,)) if isinstance(ignored, str) else frozenset(ignored)


def make_ignore_filter(
    mutation_id: str,
    prefix: str,
    project_ignores: FrozenSet[str],
    comment_ignores: CommentIgnores,
) -> CallableWrapper[bool]:
    ignored_in_project = mutation_id in project_ignores

    def allows(_node: CSTNode, context: Context) -> bool:
        if ignored_in_project:
            return False
        ignored = comment_ignores.for_context(context)
        return mutation_id not in ignored and prefix not in ignored

    return CallableWrapper(allows)
