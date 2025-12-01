package fol.term;

import fol.Substitution;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public record Function(FSymbol symbol, List<Term> args) implements Term {
    public Function {
        assert symbol.arity() == args.size();
    }
    
    @Override
    public Term applySub(Substitution substitution) {
        List<Term> newArgs = args.stream().map(t -> t.applySub(substitution)).toList();
        return new Function(symbol, newArgs);
    }

    public String name() {
        return symbol.name();
    }

    @Override
    public String toString() {
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
    public Set<Term> skolemTerms() {
        Set<Term> out = new HashSet<>();
        for (Term t : args) out.addAll(t.skolemTerms());
        return out;
    }

    @Override
    public int hashCode() {
        return ("C:" + this).hashCode();
    }

    @Override
    public boolean equals(Object other) {
        if(!(other instanceof Function otherFunc)) return false;
        return symbol.equals(otherFunc.symbol) && args.equals(otherFunc.args);
    }
}
