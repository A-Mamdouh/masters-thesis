from .formula import Formula
from .p_form import PSymbol, PForm
from .fnot import Not

true: PForm = PForm(PSymbol("T", 0), [])
false: Formula = Not(true)
