from typing import Optional

from nls_python.typing_utils import typechecked

from .substitution import Substitution, Substitutable
from .term import Term, FTerm


@typechecked
def unify(
    t1: Term, t2: Term, sub: Optional[Substitution] = None
) -> Optional[Substitution]:
    if sub is None:
        sub = Substitution()
    t1 = t1.apply_sub(sub)
    t2 = t2.apply_sub(sub)

    if t1 == t2:
        return sub
    if isinstance(t1, Substitutable):
        return _unify_var(t1, t2, sub)
    if isinstance(t2, Substitutable):
        return _unify_var(t2, t1, sub)
    if not isinstance(t1, FTerm) or not isinstance(t2, FTerm):
        return None
    f1: FTerm = t1
    f2: FTerm = t2
    if f1.symbol != f2.symbol:
        return None
    for argf1, argf2 in zip(f1.args, f2.args):
        sub = unify(argf1, argf2, sub)
        if sub is None:
            return None
    return sub


@typechecked
def _unify_var(
    var: Substitutable, t: Term, sub: Substitution
) -> Optional[Substitution]:
    if t == var:
        return sub
    if _occurs_check(var, t):
        return None
    new_sub = Substitution()
    new_sub.put(var, t)
    return sub.compose(new_sub)


@typechecked
def _occurs_check(var: Substitutable, t: Term) -> bool:
    if isinstance(t, Substitutable):
        return t == var
    if isinstance(t, FTerm):
        for arg in t.args:
            if _occurs_check(var, arg):
                return True
    return False
