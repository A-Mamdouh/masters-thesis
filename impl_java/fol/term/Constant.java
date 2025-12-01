package fol.term;

import fol.Substitution;

import java.util.HashSet;
import java.util.Set;

public record Constant(String name) implements Term {
    @Override
    public Term applySub(Substitution substitution) {
        return this;
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public Set<Variable> vars() {
        return new HashSet<>();
    }

    @Override
    public Set<Term> skolemTerms() {
        return Set.of();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Constant other && other.name == name;
    }

    @Override
    public int hashCode() {
        return ("C:" + this).hashCode();
    }
}
