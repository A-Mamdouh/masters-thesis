package fol.formula;

import fol.Substitution;
import fol.term.Term;
import fol.term.Variable;
import java.util.Set;

public record Not(Formula formula) implements Formula {
    @Override
    public Formula applySub(Substitution substitution) {
        return new Not(formula.applySub(substitution));
    }

    @Override
    public String toString() {
        return "¬" + formula;
    }

    // @Override
    // public List<Term> getTerms() {
    //     return formula.getTerms();
    // }

    @Override
    public int countLiterals() {
        return formula.countLiterals();
    }

    @Override
    public Formula toSNF() {
        return toNNF().toSNF();
    }

    @Override
    public Set<Variable> freeVars() {
        return formula.freeVars();
    }

    @Override
    public Formula toNNF() {
        return switch (formula) {
            case Predicate _ -> this;
            case Not not ->
                // Double negation: ¬¬A → A
                    not.formula.toNNF();
            case And and ->
                // De Morgan: ¬(A ∧ B) → ¬A ∨ ¬B
                // Note: Since we only have AND, we'll represent OR using NOT and AND
                // ¬A ∨ ¬B ≡ ¬(¬¬A ∧ ¬¬B)
                    new Not(new And(
                            new Not(and.left()).toNNF(),
                            new Not(and.right()).toNNF()
                    )).toNNF();
            case Forall forall ->
                // ¬∀x.P → ∃x.¬P
                    new Exists(forall.var(), forall.precondition(),
                            new Not(forall.formula()).toNNF());
            case Exists exists ->
                // ¬∃x.P → ∀x.¬P
                    new Forall(exists.var(), exists.precondition(),
                            new Not(exists.formula()).toNNF());
            default -> throw new IllegalStateException("Unexpected formula type: " + formula.getClass());
        };
    }

    @Override
    public Set<Term> skolemTerms() {
        return formula.skolemTerms();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Not(Formula formula1))) return false;
        return formula.equals(formula1);
    }

    @Override
    public int hashCode() {
        return getEqString().hashCode();
    }

    @Override
    public String getEqString() {
        return "¬" + formula.getEqString();
    }
}