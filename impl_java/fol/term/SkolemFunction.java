package fol.term;

import fol.Substitution;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

final public class SkolemFunction implements Term, Substitutable {

    private final FSymbol symbol;
    private final List<Term> args;

    public FSymbol symbol() {
        return symbol;
    }

    public List<Term> args() {
        return args;
    }

    public SkolemFunction(List<Term> args) {
        this.symbol = new FSymbol(defaultSymPrefix + getNum(), args.size());
        this.args = args;
    }

    private String getNum() {
        // Thread safe incrementation
        synchronized(SkolemFunction.class) {
            ++num;
            return "" + num;
        }
    }

    public SkolemFunction(FSymbol symbol, List<Term> args) {
        this.symbol = symbol;
        this.args = args;
    }

    public final String defaultSymPrefix = "sk_";
    private static int num = 0;

    @Override
    public Term applySub(Substitution substitution) {
        List<Term> newArgs = args.stream().map(t -> t.applySub(substitution)).toList();
        Term newFunc = new SkolemFunction(symbol, newArgs);
        return substitution.getOrDefault(this, newFunc);
    }

    @Override
    public String toString() {
        if (args.isEmpty()) return symbol.name();
        return symbol.name() + "(" + String.join(", ", args.stream().map(Object::toString).toArray(String[]::new)) + ")";
    }

    @Override
    public Set<Variable> vars() {
        return args.stream()
                .map(Term::vars)
                .reduce(new HashSet<>(), (set1, set2) -> {
                    set1.addAll(set2);
                    return set1;
                });
    }

    @Override
    public int hashCode() {
        return getEqString().hashCode();
    }

    private String getEqString() {
        return "S:" + this;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof SkolemFunction other)) return false;
        return getEqString().equals(other.getEqString());
    }

    @Override
    public Set<Term> skolemTerms() {
        return Set.of(this);
    }
}
