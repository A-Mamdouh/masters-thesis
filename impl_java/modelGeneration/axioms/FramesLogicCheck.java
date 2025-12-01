package modelGeneration.axioms;

import fol.Language;
import fol.formula.Formula;
import fol.formula.Predicate;
import fol.term.Term;
import modelGeneration.folModel.Model;

import java.util.*;

public class FramesLogicCheck implements ConsistencyCheck {


    @Override
    public boolean check(Model model) {
        Map<Term, List<RoleData>> framesInfo = new HashMap<>();
        // Group model's formulas by frames
        // for each frame. No role can be duplicated
        for(final var formula : model.getFormulas()) {
            final var maybeRole = RoleData.fromFormula(formula);
            if(maybeRole.isEmpty()) {
                continue;
            }
            final var roleData = maybeRole.get();
            if(!framesInfo.containsKey(roleData.frame())) {
                framesInfo.put(roleData.frame(), new LinkedList<>());
            }
            final var roles = framesInfo.get(roleData.frame());
            for(final var roleData2 : roles) {
                if(roleData2.roleName().equals(roleData.roleName())) {
                    return false;
                }
            }
            roles.add(maybeRole.get());
        }
        return true;
    }

    private record RoleData(Term frame, Term roleName, Term adopter) {
        public static Optional<RoleData> fromFormula(Formula formula) {
            if(formula instanceof Predicate p && p.symbol().equals(Language.Predicates.rolePSym)) {
                return Optional.of(new RoleData(p.args().get(0), p.args().get(1), p.args().get(2)));
            }
            return Optional.empty();
        }
    }

    @Override
    public ConsistencyCheck getCopy() {
        return new FramesLogicCheck();
    }
}
