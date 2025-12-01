package fol.formula;

import fol.Substitution;
import fol.term.Term;
import fol.term.Variable;
import java.util.HashSet;
import java.util.Set;

public record And(Formula left, Formula right) implements Formula {
    @Override
    public Formula applySub(Substitution substitution) {
        return new And(left.applySub(substitution), right.applySub(substitution));
    }

    @Override
    public String toString() {
        return "(" + left + " ∧ " + right + ")";
    }

    @Override
    public int countLiterals() {
        return left.countLiterals() + right.countLiterals();
    }

    // @Override
    // public List<Term> getTerms() {
    //     final var out = left.getTerms();
    //     out.addAll(right.getTerms());
    //     return out;
    // }

    @Override
    public String getEqString() {
        return "(" + left.getEqString() + " ∧ " + right.getEqString() + ")";
    }

    @Override
    public int hashCode() {
        return getEqString().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof And(Formula left1, Formula right1))) return false;
        return left.equals(left1) && right.equals(right1);
    }

    @Override
    public Set<Variable> freeVars() {
        Set<Variable> out = left.freeVars();
        out.addAll(right.freeVars());
        return out;
    }

    @Override
    public Formula toNNF() {
        // Predicates are already in NNF
        return this;
    }

    @Override
    public Formula toSNF() {
        return new And(left.toSNF(), right.toSNF());
    }

    @Override
    public Set<Term> skolemTerms() {
        Set<Term> out = new HashSet<>(left.skolemTerms());
        out.addAll(right.skolemTerms());
        return out;
    }
}