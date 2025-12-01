package modelGeneration.folModel.rules;

import fol.Substitution;
import fol.Unifier;
import fol.formula.Equals;
import fol.formula.Formula;
import modelGeneration.folModel.Model;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class EqElim implements InferenceRule {
    @Override
    public boolean apply(final Model model) {
        var applications = getRuleApplications(getEqs(model), model);
        if (applications.isEmpty()) return false;
        for(var application : applications){
            model.applySubstitution(application.sub());
            model.addRuleApplications(Set.of(application));
        }
        return true;
    }

    public Set<Equals> getEqs(Model model) {
        return model.getFormulas().stream()
                .filter(Equals.class::isInstance)
                .map(Equals.class::cast)
                .collect(Collectors.toSet());
    }

    public Set<EqElimRuleApplication> getRuleApplications(Set<Equals> eqForms, Model model) {
        Set<Formula> allProducts = new HashSet<>();
        Set<EqElimRuleApplication> applications = new HashSet<>();
        Substitution sub = model.getSubstitution();
        var modelFormulas = model.getFormulas();
        for (Equals eq : eqForms) {
            var maybeSub = Unifier.unify(eq.left(), eq.right(), model.getSubstitution());
            if(maybeSub.isEmpty())
                continue;
            final var innerSub = sub.compose(maybeSub.get());
            var emptyApplication = new EqElimRuleApplication(eq, maybeSub.get(), Set.of());
            if(model.getRuleApplications().contains(emptyApplication)){
                continue;
            }
            var out = modelFormulas.stream()
                    .map(f -> f.applySub(innerSub))
                    .filter(f -> !allProducts.contains(f) && !modelFormulas.contains(f))
                    .collect(Collectors.toSet());
            allProducts.addAll(out);
            applications.add(new EqElimRuleApplication(eq, maybeSub.get(), out));
            sub = innerSub;
        }
        return applications;
    }

    @Override
    public boolean isBranching() {
        return false;
    }

    public record EqElimRuleApplication(Equals input, Substitution sub, Set<Formula> output) implements RuleApplication {

        @Override
        public String toString() { return getString(0, ""); }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof EqElimRuleApplication(Equals input1, Substitution _, Set<Formula> _)) {
                return input.equals(input1);
            }
            return false;
        }

        @Override
        public String getString(int indentation, String delim) {
            if (indentation == 0)
                return "EqElim : " + input + " -> " + output;
            String indentString = "  ".repeat(indentation) + delim;
            return indentString + " EqElim: " + input + " -> " + output;
        }

        @Override
        public int hashCode() {
            return input.hashCode();
        }
    }
}
