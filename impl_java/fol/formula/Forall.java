package fol.formula;

import fol.Substitution;
import fol.term.Term;
import fol.term.Variable;
import java.util.Set;

public class Forall implements Formula {

    private final Variable var;
    private final Formula precondition;
    private final Formula formula;

    public Forall(Variable var, Formula precondition, Formula formula) {
        this.var = var;
        this.precondition = precondition;
        this.formula = formula;
    }

    @Override
    public int countLiterals() {
        return formula.countLiterals() + precondition.countLiterals();
    }

    public Forall(Variable var, Formula formula) {
        this.var = var;
        this.precondition = Predicate.TRUE;
        this.formula = formula;
    }

    public Variable var() {
        return var;
    }

    public Formula precondition() {
        return precondition;
    }

    public Formula formula() {
        return formula;
    }

    @Override
    public Formula applySub(Substitution substitution) {
        // Avoid substituting the bound variable
        Substitution pruned = substitution.without(var);
        return new Forall(var, precondition.applySub(pruned), formula.applySub(pruned));
    }

    public Formula apply(Term t) {
        Substitution sub = new Substitution();
        sub.put(var, t);
        return formula.applySub(sub);
    }

    public Formula applyPrecondition(Term t) {
        Substitution sub = new Substitution();
        sub.put(var, t);
        return precondition.applySub(sub);
    }

    @Override
    public String toString() {
        return String.format("∀%s:%s.%s", var, precondition, formula);
    }

    @Override
    public Set<Variable> freeVars() {
        Set<Variable> out = formula.freeVars();
        out.remove(var);
        return out;
    }

    @Override
    public Formula toNNF() {
        // Convert the inner formula to NNF
        return new Forall(var, precondition.toNNF(), formula.toNNF());
    }

    @Override
    public Formula toSNF() {
        // Universal quantifiers remain in the formula
        // Just convert the inner formula to SNF
        return new Forall(var, precondition.toSNF(), formula.toSNF());
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (!(obj instanceof Forall other)) return false;
        return apply(var).equals(other.apply(var));
    }

    @Override
    public int hashCode() {
        return getEqString().hashCode();
    }

    @Override
    public String getEqString() {
        var var = new Variable("$A$");
        return new Forall(var, applyPrecondition(var), apply(var)).toString();
    }

    @Override
    public Set<Term> skolemTerms() {
        return formula.skolemTerms();
    }

    // @Override
    // public List<Term> getTerms() {
    //     final var out = precondition.getTerms();
    //     out.add(var);
    //     out.addAll(precondition.getTerms());
    //     out.addAll(formula.getTerms());
    //     return out;
    // }

}