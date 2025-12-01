package fol.formula;

import fol.Substitution;
import fol.term.SkolemFunction;
import fol.term.Term;
import fol.term.Variable;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

public class Exists implements Formula {

    private final Variable var;
    private final Formula precondition;
    private final Formula formula;

    public Exists(Variable var, Formula precondition, Formula formula) {
        this.var = var;
        this.precondition = precondition;
        this.formula = formula;
    }

    @Override
    public int countLiterals() {
        return formula.countLiterals() + precondition.countLiterals();
    }

    public Exists(Variable var, Formula formula) {
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
        return new Exists(var, precondition.applySub(pruned), formula.applySub(pruned));
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
        return String.format("∃%s:%s.%s", var, precondition, formula);
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
        return new Exists(var, precondition.toNNF(), formula.toNNF());
    }


    @Override
    public Formula toSNF() {
        // Exists quantifiers need to be skolemized
        // First, convert the inner formula to SNF
        Formula skolemized = formula.toSNF();
        // The existential variable should be replaced with a Skolem function
        // whose arguments are all the free variables in scope
        List<Term> freeVars = new LinkedList<>(skolemized.freeVars().stream().map(var -> (Term) var).toList());
        freeVars.remove(var);
        SkolemFunction skolemTerm = new SkolemFunction(freeVars);
        // Create substitution and apply it
        Substitution sub = new Substitution();
        sub.put(var, skolemTerm);
        return skolemized.applySub(sub);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (!(obj instanceof Exists other)) return false;
        return apply(var).equals(other.apply(var));
    }

    @Override
    public int hashCode() {
        return getEqString().hashCode();
    }

    // @Override
    // public List<Term> getTerms() {
    //     final var out = precondition.getTerms();
    //     out.add(var);
    //     out.addAll(precondition.getTerms());
    //     out.addAll(formula.getTerms());
    //     return out;
    // }

    @Override
    public String getEqString() {
        var var = new Variable("$E$");
        return new Exists(var, applyPrecondition(var), apply(var)).toString();
    }

    @Override
    public Set<Term> skolemTerms() {
        return formula.skolemTerms();
    }
}