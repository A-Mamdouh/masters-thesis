package fol.formula;

import fol.Substitution;
import fol.term.Term;
import fol.term.Variable;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Predicate implements Formula {

    private final PSymbol symbol;
    private final List<Term> args;

    public Predicate(PSymbol symbol, List<Term> args) {
        assert symbol.arity() == args.size();
        this.symbol = symbol;
        this.args = args;
    }

    @Override
    public int countLiterals() {
        return 1; // Each predicate counts as one literal
    }

    // @Override
    // public List<Term> getTerms() {
    //     final List<Term> out = new LinkedList<>();
    //     out.addAll(args);
    //     return out;
    // }

    public Predicate(String symbolStr, List<Term> args) {
        this.symbol = new PSymbol(symbolStr, args.size());
        this.args = args;
    }

    public PSymbol symbol() {
        return symbol;
    }

    public List<Term> args() {
        return List.copyOf(args);
    }

    public static final Formula TRUE = new Predicate(new PSymbol("true", 0), List.of());
    public static final Formula FALSE = new Not(TRUE);

    @Override
    public Formula applySub(Substitution substitution) {
        List<Term> newArgs = args.stream().map(t -> t.applySub(substitution)).toList();
        return new Predicate(symbol, newArgs);
    }

    @Override
    public String toString() {
        if (args.isEmpty()) {
            return symbol.name();
        }
        return symbol.name() + "(" + String.join(", ", args.stream().map(Object::toString).toArray(String[]::new)) + ")";
    }

    @Override
    public Formula toNNF() {
        // Predicates are already in NNF
        return this;
    }

    @Override
    public Set<Variable> freeVars() {
        return args.stream()
                .map(Term::vars)
                .reduce(new HashSet<>(), (set1, set2) -> {
                    set1.addAll(set2);
                    return set1;
                });
    }

    @Override
    public Formula toSNF() {
        return this;
    }

    @Override
    public Set<Term> skolemTerms() {
        Set<Term> out = new HashSet<>();
        for (Term t : args) out.addAll(t.skolemTerms());
        return out;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Predicate)) return false;
        return getEqString().equals(((Predicate) obj).getEqString());
    }

    @Override
    public int hashCode() {
        return getEqString().hashCode();
    }

    @Override
    public String getEqString() {
        return toString();
    }
}
