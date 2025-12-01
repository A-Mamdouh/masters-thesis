package modelGeneration.folModel.rules;

import fol.formula.And;
import fol.formula.Formula;
import modelGeneration.folModel.Model;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class AndElim implements InferenceRule {

    @Override
    public boolean apply(final Model model) {
        var applications = getRuleApplications(getAnds(model), model);
        if (applications.isEmpty()) return false;
        model.addFormulas(
                applications.stream()
                        .map(AndElimRuleApplication::output)
                        .flatMap(Collection::stream)
                        .collect(Collectors.toSet())
        );
        model.addRuleApplications(applications.stream()
                .map(RuleApplication.class::cast)
                .collect(Collectors.toSet())
        );
        return true;
    }

    @Override
    public boolean isBranching() {
        return false;
    }

    private Set<AndElimRuleApplication> getRuleApplications(Set<And> ands, Model model) {
        Set<Formula> allProducts = new HashSet<>();
        Set<AndElimRuleApplication> applications = new HashSet<>();
        Set<Formula> modelFormulas = model.getFormulas();
        Set<RuleApplication> modelRuleApplications = model.getRuleApplications();
        for (And and : ands) {
            Set<Formula> products = new HashSet<>();
            if (!modelFormulas.contains(and.left()) && !allProducts.contains(and.left())) {
                allProducts.add(and.left());
                products.add(and.left());
            }
            if (!modelFormulas.contains(and.right()) && !allProducts.contains(and.right())) {
                allProducts.add(and.right());
                products.add(and.right());
            }
            AndElimRuleApplication application = new AndElimRuleApplication(and, products);
            if (!applications.contains(application) && !modelRuleApplications.contains(application)) {
                applications.add(application);
            }
        }
        return applications;
    }

    private Set<And> getAnds(Model model) {
        return model.getFormulas().stream()
                .filter(And.class::isInstance)
                .map(And.class::cast)
                .collect(Collectors.toSet());
    }


    public record AndElimRuleApplication(And input, Set<Formula> output) implements RuleApplication {

        @Override
        public String toString() { return getString(0, ""); }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof AndElimRuleApplication(And input1, Set<Formula> _)) {
                return input.equals(input1);
            }
            return false;
        }

        @Override
        public String getString(int indentation, String delim) {
            if (indentation == 0)
                return "AndElim : " + input + " -> " + output;
            String indentString = "  ".repeat(indentation) + delim;
            return indentString + " AndElim: " + input + " -> " + output;
        }

        @Override
        public int hashCode() {
            return input.hashCode();
        }

    }
}
