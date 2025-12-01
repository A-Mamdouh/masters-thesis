package fol.formula;

import fol.Substitution;
import fol.term.Term;
import java.util.Comparator;
import java.util.List;

public class Equals extends Predicate {
    public static final PSymbol EQ_PRED_SYM = new PSymbol("=", 2);

    private final Term left;
    private final Term right;

    public Term left() {
        return left;
    }

    public Term right() {
        return right;
    }

    public Equals(Term left, Term right) {
        super(EQ_PRED_SYM, List.of(left, right).stream().sorted(Comparator.comparing(Object::toString)).toList());
        this.left = left;
        this.right = right;
    }

    @Override
    public String toString() {
        return left.toString() + " = " + right.toString();
    }

    @Override
    public int hashCode() {
        // Since the super constructor sorts the arguments, commutativity becomes a nonissue
        return super.hashCode();
    }

    @Override
    public Formula applySub(Substitution substitution) {
        return new Equals(left.applySub(substitution), right.applySub(substitution));
    }

    @Override
    public boolean equals(Object obj) {
        // Since the super constructor sorts the arguments, commutativity becomes a nonissue
        return super.equals(obj);
    }
}
