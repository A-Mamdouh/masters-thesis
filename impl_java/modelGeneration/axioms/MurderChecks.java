package modelGeneration.axioms;

import fol.formula.Formula;
import fol.formula.Predicate;
import fol.term.Term;
import modelGeneration.folModel.Model;
import ontology.Data;

import java.util.*;

import static fol.Language.Predicates.framePSym;
import static fol.Language.Predicates.rolePSym;

public class MurderChecks implements ConsistencyCheck {
    @Override
    public boolean check(Model model) {
        // Filter murders
        for (final var formula : model.getFormulas()) {
            final var maybeMurder = Murder.fromFormula(formula, model);
            if(maybeMurder.isEmpty()) continue;
            final var murder = maybeMurder.get();
            if(murder.murderer == null) continue;
            if(murder.location == null) continue;
            // Has to be at the same location
            if(!isTermAtLocation(murder.murderer(), murder.location(), murder.situation, model))
                return false;
            // Cannot murder oneself
            if(murder.victim().equals(murder.murderer()))
                return false;
        }
        return true;
    }

    @Override
    public ConsistencyCheck getCopy() {
        return new MurderChecks();
    }

    private boolean isTermAtLocation(Term toLocate, Term location, Term situation, Model model) {

        Map<Term, List<Predicate>> framesBySituation = new HashMap<>();
        Map<Term, Predicate> frameFormulaByFrameTerm = new HashMap<>();
        Map<Term, List<Predicate>> rolesbyFrame = new HashMap<>();
        List<Predicate> locationFormulas = new LinkedList<>();

        for(final var formula : model.getFormulas()) {
            if(formula instanceof Predicate p && p.symbol().equals(framePSym)) {
                framesBySituation.computeIfAbsent(p.args().get(0), k -> new LinkedList<>()).add(p);
                frameFormulaByFrameTerm.put(p.args().get(2), p);
            }
            if(formula instanceof Predicate p && p.symbol().equals(rolePSym)) {
                rolesbyFrame.computeIfAbsent(p.args().get(0), k -> new LinkedList<>()).add(p);
            }
            if(formula instanceof Predicate p && p.symbol().equals(rolePSym)
                    && p.args().get(1).toString().equals("role_location")
                    && p.args().get(2).equals(location)
            ) {
                locationFormulas.add(p);
            }
        }

        for(final var locationFormula : locationFormulas) {
            final var frame = locationFormula.args().get(0);
            // Only include beAt or attend events
            final var frameFormula = frameFormulaByFrameTerm.get(frame);
            if(frameFormula == null)
                throw new IllegalStateException("Frame formula not found for frame term " + frame);
            final var frameName = frameFormula.args().get(1);
            if(!List.of(Data.Verbs.beAt, Data.Verbs.attend).contains(frameName.toString()))
                continue;
            final var situationTerm = frameFormula.args().get(0);
            if(!situationTerm.equals(situation))
                continue;
            for(final var roleFormula : rolesbyFrame.getOrDefault(frame, Collections.emptyList())) {
                final var roleAdopter = roleFormula.args().get(2);
                if(roleAdopter.equals(toLocate))
                    return true;
            }
        }

        return false;
//
//        // Find beAt frames
//        Set<Term> beAtFrames = new HashSet<>();
//        for(final var formula : model.getFormulas()) {
//            if (formula instanceof Predicate p && p.symbol().equals(framePSym)
//                    && p.args().get(0).equals(situation) && p.args().get(1).toString().equals(Data.Verbs.beAt)
//            ) {
//                beAtFrames.add(p.args().get(2));
//            }
//        }
//
//        if(beAtFrames.isEmpty())
//            return false;
//        for(final var frame : beAtFrames) {
//            boolean locationFound = false;
//            boolean goalFound = false;
//            for(final var formula : model.getFormulas()) {
//                if (formula instanceof Predicate p && p.symbol().equals(rolePSym)
//                        && p.args().get(1).toString().equals("role_location") && p.args().get(0).equals(frame)
//                ) {
//                    locationFound |= p.args().get(2).equals(location);
//                }
//                if (formula instanceof Predicate p && p.symbol().equals(rolePSym)
//                        && p.args().get(1).toString().equals("role_attendee") && p.args().get(0).equals(frame)
//                ) {
//                    goalFound |= p.args().get(2).equals(toLocate);
//                }
//                if(locationFound && goalFound)
//                    break;
//            }
//            if(locationFound && goalFound)
//                return true;
//        }
//        return false;
    }

    private record Murder(Term situation, Term frame, Term murderer, Term victim, Term location) {

        private static Optional<Murder> fromFormula(Formula form, Model model) {
            if (!(form instanceof Predicate)) return Optional.empty();
            final var predicate = (Predicate) form;
            if (!predicate.symbol().equals(framePSym)) {
                return Optional.empty();
            }
            if (!predicate.args().get(1).toString().equals(Data.Verbs.murder)) {
                return Optional.empty();
            }
            final var situation = predicate.args().get(0);
            final var frame = predicate.args().get(2);
            // Search for the murderer, victim and location
            Term location = null;
            Term murderer = null;
            Term victim = null;
            for (final var formula2 : model.getFormulas()) {
                if (formula2 instanceof Predicate p && p.symbol().equals(rolePSym)
                        && p.args().get(1).toString().equals("role_location") && p.args().get(0).equals(frame)
                ) {
                    // TODO: Check if the location is known
                    location = p.args().get(2);
                }

                if (formula2 instanceof Predicate p && p.symbol().equals(rolePSym)
                        && p.args().get(1).toString().equals("role_murderer") && p.args().get(0).equals(frame)
                ) {
                    // TODO: Check if the murderer is known
                    murderer = p.args().get(2);
                }

                if (formula2 instanceof Predicate p && p.symbol().equals(rolePSym)
                        && p.args().get(1).toString().equals("role_victim") && p.args().get(0).equals(frame)
                ) {
                    // TODO: Check if the victim is known
                    victim = p.args().get(2);
                }
            }

            return Optional.of(new Murder(situation, frame, murderer, victim, location));
        }

    }
}
