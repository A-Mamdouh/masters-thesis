package fol.term;

import fol.Substitution;

import java.util.Set;

public sealed interface Term permits Constant, Function, SkolemFunction, Variable {
    Term applySub(Substitution substitution);
    Set<Variable> vars();

    /**
     * Get a set of skolem terms (constants and applied function symbols) inside the term.
     *
     * @return a set of skolem terms inside the term
     */
    Set<Term> skolemTerms();
}
