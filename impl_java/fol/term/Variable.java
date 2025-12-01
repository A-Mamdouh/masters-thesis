package fol.term;

import fol.Substitution;

import java.util.HashSet;
import java.util.Set;

public record Variable(String name) implements Term, Substitutable {

    static private int num = 0;

    public static Variable make() {
        String name;
        synchronized(SkolemFunction.class) {
            ++num;
             name = "_V" + num;
        }
        return new Variable(name);
    }

    @Override
    public Term applySub(Substitution substitution) {
        return substitution.getOrDefault(this, this);
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public Set<Variable> vars() {
        return new HashSet<>(Set.of(this));
    }

    @Override
    public Set<Term> skolemTerms() {
        return Set.of();
    }

    @Override
    public int hashCode() {
        return ("V:" + this).hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Variable other)) return false;
        return name.equals(other.name);
    }
}
