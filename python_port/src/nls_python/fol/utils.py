from __future__ import annotations

from typing import TYPE_CHECKING

from nls_python.typing_utils import typechecked

if TYPE_CHECKING:
    from .formula.formula import Formula


@typechecked
def is_literal(f: "Formula") -> bool:
    from .formula.p_form import PForm
    from .formula.fnot import Not
    from .formula.equals import Equals

    if isinstance(f, (PForm, Equals)):
        return True
    return isinstance(f, Not) and is_literal(f.inner)
