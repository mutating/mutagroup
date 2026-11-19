from typing import Dict

from pristan import slot

from mutagroup import MutationsCollection


# TODO: Remove the empty parentheses once the pristan issue (https://github.com/mutating/pristan/issues/31) is fixed
@slot()
def mutations() -> Dict[str, MutationsCollection]:
    # Pristan recognizes the empty literal and supplies the result.
    return {}  # pragma: no cover
