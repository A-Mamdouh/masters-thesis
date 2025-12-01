package fol.formula;

import fol.Substitution;
import fol.term.Term;
import fol.term.Variable;
import java.util.Set;

public interface Formula {
    Formula applySub(Substitution substitution);

    /**
     * Transforms the formula to Skolem normal form
     * @return Skolem-ized formula
     */
    Formula toSNF();

    /**
     * Get a set of the current free variables inside the formula
     * @return set of free variables in the current formula
     */
    Set<Variable> freeVars();

    /**
     * Convert formula to negated normal form (needed for SNF)
     * @return the formula in NNF
     */
    Formula toNNF();

    /**
     * Get a set of skolem terms (constants and applied function symbols) inside the formula.
     *
     * @return a set of skolem terms inside the formula
     */
    Set<Term> skolemTerms();

    String getEqString();

    int countLiterals();

    // List<Term> getTerms();

}
