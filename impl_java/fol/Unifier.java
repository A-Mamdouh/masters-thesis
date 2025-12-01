package fol;

import java.util.Optional;

import fol.term.Function;
import fol.term.Substitutable;
import fol.term.Term;

public class Unifier {
    public static Optional<Substitution> unify(Term t1, Term t2) {
        return unify(t1, t2, new Substitution());
    }

    public static Optional<Substitution> unify(Term t1, Term t2, Substitution theta) {
        t1 = t1.applySub(theta);
        t2 = t2.applySub(theta);

        if (t1.equals(t2)) {
            return Optional.of(theta);
        } else if (t1 instanceof Substitutable var) {
            return unifyVar(var, t2, theta);
        } else if (t2 instanceof Substitutable var) {
            return unifyVar(var, t1, theta);
        } else if (t1 instanceof Function f1 && t2 instanceof Function f2) {
            if (!f1.name().equals(f2.name()) || f1.args().size() != f2.args().size()) {
                return Optional.empty();
            }
            Substitution current = theta;
            for (int i = 0; i < f1.args().size(); i++) {
                var res = unify(f1.args().get(i), f2.args().get(i), current);
                if (res.isEmpty()) return Optional.empty();
                current = res.get();
            }
            return Optional.of(current);
        } else {
            return Optional.empty();
        }
    }

    private static Optional<Substitution> unifyVar(Substitutable var, Term term, Substitution theta) {
        if (term.equals(var)) {
            return Optional.of(theta);
        } else if (occursCheck(var, term)) {
            return Optional.empty();
        } else {
            Substitution sigma = new Substitution();
            sigma.put(var, term);
            return Optional.of(theta.compose(sigma));
        }
    }

    private static boolean occursCheck(Substitutable var, Term term) {
        if (term instanceof Substitutable v) {
            return v.equals(var);
        } else if (term instanceof Function f) {
            for (Term arg : f.args()) {
                if (occursCheck(var, arg)) return true;
            }
        }
        return false;
    }
}
