package modelGeneration.folModel.rules;

import fol.formula.And;
import fol.formula.Formula;
import fol.formula.Not;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import modelGeneration.folModel.Model;

public class OrElim implements InferenceRule {

    @Override
    public boolean apply(Model model) {
        var or = getOr(model);
        if(or.isEmpty()) return false;
        var maybeApplication = getRuleApplication(or.get(), model);
        if (maybeApplication.isEmpty()) return false;
        var application = maybeApplication.get();
        model.addExtensions(Map.of(application, application.outputExtensions()));
        model.addRuleApplications(Set.of(application));
        return true;
    }

    @Override
    public boolean isBranching() {
        return true;
    }

    private Optional<Not> getOr(Model model) {
        return model.getFormulas().stream()
                .filter(f -> f instanceof Not && ((Not) f).formula() instanceof And)
                .map(Not.class::cast).findAny();
    }

    private Optional<OrElimRuleApplication> getRuleApplication(Not or, Model model) {
        Set<RuleApplication> modelRuleApplications = model.getRuleApplications();
        // Empty rule application marks that the formula was already expanded
        OrElimRuleApplication emptyApplication = new OrElimRuleApplication(or, Set.of());
        if (modelRuleApplications.contains(emptyApplication)) {
            return Optional.empty();
        }
        Formula left = ((And) or.formula()).left();
        // Add negation and simplify double negation
        left = left instanceof Not(Formula newLeft) ? newLeft : new Not(left);
        Formula right = ((And) or.formula()).right();
        right = right instanceof Not(Formula newRight) ? newRight : new Not(right);
        // Create left and right extensions
        Model leftExtension = new Model(model.getFormulas(), model.getSentenceDepth(), model.getIndividuals(), Map.of(), model.getRuleApplications(), model.getParent());
        leftExtension.addFormulas(Set.of(left));
        leftExtension.setParent(model);
        Model rightExtension = new Model(model.getFormulas(), model.getSentenceDepth(), model.getIndividuals(), Map.of(), model.getRuleApplications(), model.getParent());
        rightExtension.addFormulas(Set.of(right));
        rightExtension.setParent(model);
        OrElimRuleApplication application = new OrElimRuleApplication(or, Set.of(leftExtension, rightExtension));
        leftExtension.addRuleApplications(Set.of(emptyApplication));
        rightExtension.addRuleApplications(Set.of(emptyApplication));
        return Optional.of(application);
    }

    public record OrElimRuleApplication(Not input, Set<Model> outputExtensions) implements RuleApplication {
        @Override
        public String toString() {
            return getString(0, "*");

        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof OrElimRuleApplication(Not otherInput, Set<Model> _)) {
                return this.input.equals(otherInput);
            }
            return false;
        }

        @Override
        public String getString(int indentation, String delim) {
            if(indentation == 0) {
                StringBuilder sb = new StringBuilder("OrElim : ");
                sb.append(input).append(" -> \n");
                for (Model model : outputExtensions) {
                    sb.append(model.createString(1, delim)).append("\n");
                }
                return sb.toString();
            }
            String outerIndent = "  ".repeat(Math.max(0, indentation)) + delim + " ";
            StringBuilder sb = new StringBuilder(outerIndent)
                    .append("OrElim : ")
                    .append(input);
            if (!outputExtensions.isEmpty()) {
                sb.append(" -> \n");
            }
            for (Model model : outputExtensions) {
                sb.append(model.createString(indentation + 1, delim)).append("\n");
            }
            return sb.toString();
        }

        @Override
        public int hashCode() {
            return input.hashCode();
        }
    }
}
