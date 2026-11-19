import sys
from pathlib import Path
from subprocess import run
from textwrap import dedent


def test_public_api_can_discover_and_apply_mutations() -> None:
    """Discover and apply standalone and combined mutations in a fresh interpreter."""
    script = dedent("""
        from cstvis import Changer
        from libcst import Integer
        from mutagroup import Context, Mutation, MutationsCollection, MutationsSuperCollection

        pack = MutationsCollection('I')

        @pack.mutation(13)
        def replace(node: Integer, context: Context) -> Integer:
            print(context.meta['mutation_name'])
            return node.with_changes(value='42')

        if not isinstance(replace, Mutation):
            raise AssertionError('The public Mutation type did not match the registered handle.')

        def allow(_node: Integer) -> bool:
            return True

        pack.filter(allow)
        replace.filter(allow)
        source = 'answer = 0'
        changers = (
            Changer(source, collector=pack.collector()),
            MutationsSuperCollection([pack]).get_changer(source),
        )
        for changer in changers:
            (target,) = changer.iterate_coordinates()
            print(target.converter_id)
            print(changer.apply_coordinate(target))
    """)
    result = run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['I-13', 'replace', 'answer = 42'] * 2
