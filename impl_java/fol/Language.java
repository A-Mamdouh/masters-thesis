package fol;

import fol.formula.PSymbol;
import fol.formula.Predicate;
import fol.term.Constant;
import fol.term.Term;
import java.util.List;

public class Language {
    public static class Predicates {
        public static final PSymbol semTypePSym = new PSymbol("semType", 2);
        public static final PSymbol rolePSym = new PSymbol("role", 3);
        public static final PSymbol framePSym = new PSymbol("frame", 3);
        public static final PSymbol situationPSym = new PSymbol("situation", 2);


        public static final Predicate semType(Term ind, Term semType) {
            return new Predicate(semTypePSym, List.of(ind, semType));
        }

        public static final Predicate role(Term frame, Term roleName, Term roleAdopter) {
            return new Predicate(rolePSym, List.of(frame, roleName, roleAdopter));
        }

        public static final Predicate frame(Term situation, Term frameName, Term adopter) {
            return new Predicate(framePSym, List.of(situation, frameName, adopter));
        }

        public static final Predicate situation(Term situationName, Term adopter) {
            return new Predicate(situationPSym, List.of(situationName, adopter));
        }

        public static final Predicate semType(String ind, String semType) {
            return semType(new Constant(ind), new Constant(semType));
        }
    }
}
