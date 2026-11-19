# mutagroup

`mutagroup` defines mutation packs for a mutation-testing engine. A pack registers
LibCST mutations and filters and supplies a `cstvis.Collector` for discovery and
application of individual mutations. A supercollection runs multiple packs over
the same source in a single coordinate discovery pass.

## Define a pack

```python
from libcst import Add, Subtract
from mutagroup import Context, MutationsCollection

mutations = MutationsCollection('I')


@mutations.mutation(13)
def change_add(node: Add) -> Subtract:
    return Subtract(
        whitespace_before=node.whitespace_before,
        whitespace_after=node.whitespace_after,
    )


@mutations.filter
def formatted_adds(node: Add) -> bool:
    return not node.whitespace_before.empty


@change_add.filter
def skip_generated(_node: Add, context: Context) -> bool:
    return context.comment != ' generated'
```

`change_add.id` is `I-13`. Use `@mutations.mutation` to take the lowest free
positive number at registration time. Explicit numbers keep public IDs stable
as a pack evolves. Registering a number twice in the same pack raises an error.

Mutations and both kinds of filters accept a node and an optional `Context`.
cstvis resolves the first parameter's annotation. Matching collection filters
run at most once per node, before bound filters. Bound filters apply to their
own mutation and matching node types. A false result suppresses the corresponding
coordinates.

When cstvis calls a mutation or one of its bound filters, `context.meta` includes
`mutation_id` and `mutation_name`. These are the registered ID and the mutation
handler's Python name. Custom metadata can use other keys.

## Run one or more packs

```python
from mutagroup import MutationsSuperCollection

packs = MutationsSuperCollection([mutations], metacode_key='muta')
source = 'a = 1 + 2 # muta: ignore[I-13]\nb = 3 + 4\n'
changer = packs.get_changer(source)
for coordinate in changer.iterate_coordinates():
    print(coordinate.converter_id, changer.apply_coordinate(coordinate))
```

Each `.get_changer(source)` call snapshots the registered mutations and filters.
`iterate_coordinates()` discovers candidates from all packs in source order,
with each pack's filters scoped to that pack. `apply_coordinate()` produces one
mutant of the original source. Pack prefixes must be unique within a
supercollection.

The default metacode key is `muta`. On the line where a node starts,
`# muta: ignore[I-13]` and `# muta: ignore["I-13"]` suppress that mutation.
`# muta: ignore[I]` suppresses mutations from pack `I`. A directive can list
multiple IDs, as in `# muta: ignore[I-13, J]`. The short `# muta ignore I-13`
form also works on its own.

Pass project-wide mutation IDs to `packs.get_changer(source, ignored=ids)`. For a
standalone pack, use `Changer(source, collector=mutations.collector(ignored=ids))`
from cstvis.

## Load packs from plugins

Define a pack in `example_mutations/plugin.py` in your plugin package. Use
`@mutations.plugin` to register a function that takes no arguments and returns
the pack.

```python
from libcst import Integer
from mutagroup import MutationsCollection, mutations

pack = MutationsCollection('N')


@pack.mutation(1)
def replace_integer(node: Integer) -> Integer:
    return node.with_changes(value='0')


@mutations.plugin
def number_pack() -> MutationsCollection:
    return pack
```

In the plugin package's `pyproject.toml`, add an entry point that tells Pristan
which module to import.

```toml
[project.entry-points.pristan]
example_mutations = 'example_mutations.plugin'
```

Run `python -m pip install .` from the plugin package's root in the Python
environment that runs `mutagroup`. `get_mutations()` discovers the installed
plugins and collects their packs into a supercollection.

```python
from mutagroup import MutationsSuperCollection

packs = MutationsSuperCollection.get_mutations()
changer = packs.get_changer('answer = 42')
for coordinate in changer.iterate_coordinates():
    print(coordinate.converter_id, changer.apply_coordinate(coordinate))
```
