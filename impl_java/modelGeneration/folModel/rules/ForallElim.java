package modelGeneration.folModel.rules;

import fol.formula.Forall;
import fol.formula.Formula;
import fol.term.Term;
import modelGeneration.Salient;
import modelGeneration.folModel.Model;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class ForallElim implements InferenceRule {
    @Override
    public boolean apply(Model model) {
        var applications = getRuleApplications(getForalls(model), model);
        if (applications.isEmpty()) return false;
        for (var application : applications) {
            model.addFormulas(new HashSet<>(application.outputFormulas().values()));
        }
        model.addRuleApplications(applications.stream()
                .map(RuleApplication.class::cast)
                .collect(Collectors.toSet()));
        return true;
    }

    @Override
    public boolean isBranching() {
        return false;
    }

    private Set<Forall> getForalls(Model model) {
        return model.getFormulas().stream()
                .filter(Forall.class::isInstance)
                .map(Forall.class::cast)
                .collect(Collectors.toSet());
    }

    private Set<ForallElimRuleApplication> getRuleApplications(Set<Forall> foralls, Model model) {
        Set<ForallElimRuleApplication> applications = new java.util.HashSet<>();
        Set<Formula> allProducts = new HashSet<>();
        Set<Formula> modelFormulas = model.getFormulas();
        var modelRuleApplications = model.getRuleApplications();
        var individuals = model.getIndividuals();
        for (Forall forall : foralls) {
            Map<Term, Formula> applicationOutput = new HashMap<>();
            for (var individual : individuals) {
                var precondition = forall.applyPrecondition(individual.obj());
                if(!model.getFormulas().contains(precondition))
                    continue;
                Formula formula = forall.apply(individual.obj());
                if (!modelFormulas.contains(formula) && !allProducts.contains(formula)) {
                    applicationOutput.put(individual.obj(), formula);
                    allProducts.add(formula);
                    model.setIndividualsSalience(Set.of(new Salient<>(individual.obj(), Salient.FULL)));
                }
            }
            if (applicationOutput.isEmpty())
                continue;
            var application = new ForallElimRuleApplication(forall, applicationOutput);
            if (!modelRuleApplications.contains(application))
                applications.add(application);
        }
        return applications;
    }

    public record ForallElimRuleApplication(Forall input,
                                            Map<Term, Formula> outputFormulas) implements RuleApplication {
        @Override
        public String getString(int indentation, String delim) {
            if (indentation == 0)
                return "∀Elim : " + input + " -> " + outputFormulas;
            String indentString = "  ".repeat(indentation) + delim;
            return indentString + " ∀Elim : " + input + " -> " + outputFormulas;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof ForallElimRuleApplication(Forall otherInput, Map<Term, Formula> outputMap)) {
                return input.equals(otherInput) && outputFormulas.keySet().containsAll(outputMap.keySet());
            }
            return false;
        }

        @Override
        public int hashCode() {
            return (input.toString() + "$" + outputFormulas).hashCode();
        }

        @Override
        public String toString() {
            return getString(0, "");
        }
    }

}
