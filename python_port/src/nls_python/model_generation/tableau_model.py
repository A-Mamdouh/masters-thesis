from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set, TYPE_CHECKING

from nls_python.typing_utils import typechecked

from nls_python.fol.formula import Formula
from nls_python.fol.formula.equals import Equals
from nls_python.fol.formula.exists import Exists
from nls_python.fol.formula.fand import And
from nls_python.fol.formula.fnot import Not
from nls_python.fol.substitution import Substitution
from nls_python.fol.term import Term
from nls_python.fol.unification import unify

from .salient import Salient
from .narratives import ModelNarrative, NarrativeStep

if TYPE_CHECKING:  # pragma: no cover
    from .consistency import ConsistencyCheck
    from .inference_rules.base import RuleApplication as RuleApplicationType
else:  # pragma: no cover
    RuleApplicationType = Any
    ConsistencyCheck = Any


@typechecked
@dataclass(eq=False)
class TableauModel:
    formulas: Set[Formula] = field(default_factory=set)
    individuals: Set[Salient[Term]] = field(default_factory=set)
    extensions: Dict[RuleApplicationType, Set["TableauModel"]] = field(
        default_factory=dict
    )
    rule_applications: Set[RuleApplicationType] = field(default_factory=set)
    substitution: Substitution = field(default_factory=Substitution)
    sentence_depth: int = 0
    parent: Optional["TableauModel"] = None
    _complete: bool = False
    _id: int = field(init=False)
    narrative: ModelNarrative = field(default_factory=ModelNarrative)

    _id_counter: ClassVar[int] = 0
    _id_lock: ClassVar[Lock] = Lock()

    def __post_init__(self) -> None:
        with TableauModel._id_lock:
            TableauModel._id_counter += 1
            object.__setattr__(self, "_id", TableauModel._id_counter)

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TableauModel) and self._id == other._id

    @property
    def id(self) -> int:
        return self._id

    def copy(self) -> "TableauModel":
        new_model = TableauModel(
            formulas=set(self.formulas),
            individuals={ind.copy() for ind in self.individuals},
            extensions={app: set(models) for app, models in self.extensions.items()},
            rule_applications=set(self.rule_applications),
            substitution=self.substitution.copy(),
            sentence_depth=self.sentence_depth,
            parent=self.parent,
            _complete=self._complete,
            narrative=self.narrative,
        )
        return new_model

    def branch_copy(self) -> "TableauModel":
        branch = TableauModel(
            formulas=set(self.formulas),
            individuals={ind.copy() for ind in self.individuals},
            extensions={},
            rule_applications=set(self.rule_applications),
            substitution=self.substitution.copy(),
            sentence_depth=self.sentence_depth,
            parent=self,
            narrative=self.narrative,
        )
        return branch

    def clip(self) -> "TableauModel":
        clipped = self.copy()
        clipped.parent = None
        clipped._complete = self._complete
        return clipped

    def is_complete(self) -> bool:
        return self._complete

    def complete(self, new_value: bool = True) -> None:
        self._complete = new_value

    def get_formulas(self) -> Set[Formula]:
        return self.formulas

    def add_formulas(self, formulas: Set[Formula]) -> None:
        if not formulas:
            return
        self.formulas.update(formulas)
        for formula in list(formulas):
            self.formulas.add(formula.apply_sub(self.substitution))

    def add_rule_applications(self, applications: Set[RuleApplicationType]) -> None:
        if not applications:
            return
        self.rule_applications.update(applications)

    def get_rule_applications(self) -> Set[RuleApplicationType]:
        return self.rule_applications

    def get_extensions(self) -> Dict[RuleApplicationType, Set["TableauModel"]]:
        return self.extensions

    def add_extensions(
        self, extensions: Dict[RuleApplicationType, Set["TableauModel"]]
    ) -> None:
        for application, models in extensions.items():
            stored = self.extensions.setdefault(application, set())
            stored.update(models)

    def get_substitution(self) -> Substitution:
        return self.substitution.copy()

    def apply_substitution(self, substitution: Substitution) -> None:
        updated_formulas: Set[Formula] = set()
        for formula in list(self.formulas):
            updated_formulas.add(formula.apply_sub(substitution))
        self.formulas.update(updated_formulas)
        self.substitution = self.substitution.compose(substitution)

    def set_individuals_salience(self, new_salience: Iterable[Salient[Term]]) -> None:
        for salient in new_salience:
            if salient in self.individuals:
                self.individuals.remove(salient)
            self.individuals.add(salient)

    def add_salient_individuals(self, individuals: Set[Salient[Term]]) -> None:
        self.individuals.update(individuals)

    def add_individuals(
        self, individuals: Set[Term], salience: float = Salient.FULL
    ) -> None:
        for individual in individuals:
            self.individuals.add(Salient(individual, salience))

    def get_individuals(self) -> Set[Salient[Term]]:
        return self.individuals

    def set_parent(self, parent: Optional["TableauModel"]) -> None:
        self.parent = parent

    def add_extensions_from_models(
        self, models: Set["TableauModel"], application: RuleApplicationType
    ) -> None:
        self.extensions[application] = models

    def set_sentence_depth(self, depth: int) -> None:
        self.sentence_depth = depth

    def get_sentence_depth(self) -> int:
        return self.sentence_depth

    def get_narrative(self) -> ModelNarrative:
        return self.narrative

    def set_story_key(self, story_key: str | None) -> None:
        self.narrative = ModelNarrative(self.narrative.steps, story_key=story_key)

    def append_narrative_step(self, step: NarrativeStep) -> None:
        self.narrative = self.narrative.append(step)

    def create_string(self, indentation: int = 0, delim: str = "*") -> str:
        outer_indent = "  " * max(indentation, 0) + delim + " "
        nested_indent = "  " * max(indentation + 1, 0) + delim + " "
        lines: List[str] = [f"Model #{self._id} (depth={self.sentence_depth}):"]
        if self.formulas:
            sorted_formulas = sorted(
                self.formulas,
                key=lambda formula: (
                    _count_literals(formula),
                    _negation_depth(formula),
                    str(formula),
                ),
            )
            lines.append(f"{outer_indent}Formulas:")
            lines.extend(
                f"{nested_indent}{formula}" for formula in map(str, sorted_formulas)
            )
        if self.individuals:
            sorted_inds = sorted(
                self.individuals,
                key=lambda salient: (-salient.salience, str(salient.obj)),
            )
            lines.append(f"{outer_indent}Individuals:")
            lines.extend(
                f"{nested_indent}{salient.obj} (salience={salient.salience:.3f})"
                for salient in sorted_inds
            )
        if self.rule_applications:
            lines.append(f"{outer_indent}Rule applications:")
            sorted_rules = sorted(
                self.rule_applications,
                key=lambda app: (app.__class__.__name__, str(app)),
            )
            for application in sorted_rules:
                lines.append(application.describe(indentation + 1, delim))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.create_string()

    def check_consistency(
        self, consistency_checks: Iterable["ConsistencyCheck"]
    ) -> bool:
        """Mirror the Java-side consistency contract."""
        for formula in list(self.formulas):
            if isinstance(formula, Not):
                inner = formula.inner
                if inner in self.formulas:
                    return False
                if isinstance(inner, Equals):
                    if inner.left == inner.right:
                        return False
            if isinstance(formula, Equals):
                maybe_sub = unify(formula.left, formula.right, self.substitution)
                if maybe_sub is None:
                    return False
                self.substitution = maybe_sub
        for check in consistency_checks:
            if not check.check(self):
                return False
        return True


def _count_literals(formula: Formula) -> int:
    if isinstance(formula, And):
        return _count_literals(formula.left) + _count_literals(formula.right)
    if isinstance(formula, Exists):
        return _count_literals(formula.inner) + _count_literals(formula.precondition)
    if isinstance(formula, Not):
        return 1 + _count_literals(formula.inner)
    return 1


def _negation_depth(formula: Formula) -> int:
    depth = 0
    current = formula
    while isinstance(current, Not):
        depth += 1
        current = current.inner
    return depth
