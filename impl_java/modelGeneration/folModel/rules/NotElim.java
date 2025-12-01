package modelGeneration.folModel.rules;

import fol.formula.Formula;
import fol.formula.Not;
import modelGeneration.folModel.Model;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class NotElim implements InferenceRule {

    @Override
    public boolean apply(Model model) {
        Set<NotElimRuleApplication> applications = getRuleApplications(getDoubleNots(model), model);
        if (applications.isEmpty()) return false;
        model.addFormulas(
                applications.stream()
                        .map(NotElimRuleApplication::output)
                        .collect(Collectors.toSet())
        );
        model.addRuleApplications(applications.stream()
                .map(a -> (RuleApplication) a)
                .collect(Collectors.toSet())
        );
        return true;
    }

    @Override
    public boolean isBranching() {
        return false;
    }

    private Set<NotElimRuleApplication> getRuleApplications(Set<Not> doubleNots, Model model) {
        Set<Formula> allProducts = new HashSet<>();
        Set<NotElimRuleApplication> applications = new HashSet<>();
        Set<Formula> modelFormulas = model.getFormulas();
        for (Not doubltNot : doubleNots) {
            Formula innerFormula = ((Not) doubltNot.formula()).formula();
            if (!modelFormulas.contains(innerFormula) && !allProducts.contains(innerFormula)) {
                allProducts.add(innerFormula);
                applications.add(new NotElimRuleApplication(doubltNot, innerFormula));
            }
        }
        return applications;
    }

    private Set<Not> getDoubleNots(Model model) {
        return model.getFormulas().stream()
                .filter(formula -> formula instanceof Not && ((Not) formula).formula() instanceof Not)
                .map(formula -> (Not) formula)
                .collect(Collectors.toSet());
    }

    public record NotElimRuleApplication(Not input, Formula output) implements RuleApplication {

        @Override
        public String toString() {
            return getString(0, "");
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof NotElimRuleApplication(Not input1, Formula _)) {
                return input.equals(input1);
            }
            return false;
        }

        @Override
        public String getString(int indentation, String delim) {
            if (indentation == 0)
                return "NotElim : " + input + " -> " + output;
            String indentString = "  ".repeat(indentation) + delim;
            return indentString + " NotElim: " + input + " -> " + output;
        }

        @Override
        public int hashCode() {
            return input.hashCode();
        }
    }
}
