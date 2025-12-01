import importlib
import importlib.util
from typing import Callable, TypeVar

T = TypeVar("T")

_typeguard_typechecked: Callable[[T], T] | None = None
_spec = importlib.util.find_spec("typeguard")
if _spec:
    typeguard_module = importlib.import_module("typeguard")
    _typeguard_typechecked = getattr(typeguard_module, "typechecked", None)


def typechecked(obj: T) -> T:
    if callable(_typeguard_typechecked):
        return _typeguard_typechecked(obj)  # type: ignore[misc]
    return obj


__all__ = ["typechecked"]
