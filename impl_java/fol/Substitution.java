package fol;

import fol.term.Substitutable;
import fol.term.Term;

import java.util.HashMap;
import java.util.Map;

public class Substitution {
    private final Map<Substitutable, Term> map;

    public Substitution() {
        this.map = new HashMap<>();
    }

    private Substitution(Map<Substitutable, Term> map) {
        this.map = map;
    }

    public Term getOrDefault(Substitutable var, Term defaultTerm) {
        return map.getOrDefault(var, defaultTerm);
    }

    public void put(Substitutable var, Term term) {
        map.put(var, term);
    }

    public Substitution compose(Substitution other) {
        Map<Substitutable, Term> newMap = new HashMap<>();
        for (var e : other.map.entrySet()) {
            newMap.put(e.getKey(), e.getValue().applySub(this));
        }
        for (var e : map.entrySet()) {
            newMap.putIfAbsent(e.getKey(), e.getValue());
        }
        return new Substitution(newMap);
    }

    public Substitution without(Substitutable var) {
        Map<Substitutable, Term> newMap = new HashMap<>(map);
        newMap.remove(var);
        return new Substitution(newMap);
    }

    @Override
    public String toString() {
        return map.toString();
    }


    public Substitution copy() {
        return new Substitution(new HashMap<>(map));
    }


}