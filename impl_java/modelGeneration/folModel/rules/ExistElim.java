package modelGeneration.folModel.rules;

import fol.formula.Exists;
import fol.term.SkolemFunction;
import fol.term.Term;
import modelGeneration.Salient;
import modelGeneration.folModel.Model;

import java.util.*;
import java.util.stream.Collectors;

public class ExistElim implements InferenceRule {

    @Override
    public boolean apply(Model model) {
        var e = getExist(model);
        if(e.isEmpty()) return false;
        var maybeApplication = getRuleApplication(e.get(), model);
        if (maybeApplication.isEmpty()) return false;
        var application = maybeApplication.get();
        Set<Model> extensions = application.outputExtensions().values()
                .stream()
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toSet());
        model.addExtensions(Map.of(application, extensions));
        model.addRuleApplications(Set.of(application));
        return true;
    }

    @Override
    public boolean isBranching() {
        return true;
    }

    private Optional<Exists> getExist(Model model) {
        return model.getFormulas().stream()
                .filter(Exists.class::isInstance)
                .map(Exists.class::cast)
                .filter(e -> {
                    // Check that the formula wasn't dispatched already
                    var emptyApplication = new ExistsElimRuleApplication(e, Map.of());
                    return !model.getRuleApplications().contains(emptyApplication);
                })
                .findAny();
    }

    private Optional<ExistsElimRuleApplication> getRuleApplication(Exists exists, Model model) {
        var modelRuleApplications = model.getRuleApplications();
        var individuals = model.getIndividuals();
        var emptyApplication = new ExistsElimRuleApplication(exists, Map.of());
        if (modelRuleApplications.contains(emptyApplication))
            return Optional.empty();
        var branches = new HashMap<Term, Optional<Model>>();
        // Create branches for each individual
        for (var individual : individuals.stream().sorted(Comparator.reverseOrder()).toList()) {
            var precondition = exists.applyPrecondition(individual.obj());
            // Skip incomplete preconditions
            if (!model.getFormulas().contains(precondition))
                continue;
            var formula = exists.apply(individual.obj());
            var branchModel = new Model(model.getFormulas(), model.getSentenceDepth(), model.getIndividuals(), model.getExtensions(), model.getRuleApplications(), model.getParent());
            var application = new ExistsElimRuleApplication(exists, Map.of(individual.obj(), Optional.empty()));
            branchModel.addFormulas(Set.of(formula));
            branchModel.addRuleApplications(Set.of(application));
            branchModel.setIndividualsSalience(Set.of(new Salient<>(individual.obj(), Salient.FULL)));
            branches.put(individual.obj(), Optional.of(branchModel));
        }

        // Create new branch with witness
        var witness = new SkolemFunction(exists.freeVars().stream().map(var -> (Term) var).toList());
        var branchFormula = exists.apply(witness);
        var witnessPrecondition = exists.applyPrecondition(witness);
        var witnessModel = new Model(model.getFormulas(), model.getSentenceDepth(), model.getIndividuals(), model.getExtensions(), model.getRuleApplications(), model);
        var application = new ExistsElimRuleApplication(exists, Map.of(witness, Optional.empty()));
        witnessModel.addFormulas(Set.of(branchFormula, witnessPrecondition));
        witnessModel.addIndividuals(Set.of(witness), Salient.FULL);
        witnessModel.addRuleApplications(Set.of(application));
        branches.put(witness, Optional.of(witnessModel));
        return Optional.of(new ExistsElimRuleApplication(exists, branches));
    }

    public record ExistsElimRuleApplication(Exists input,
                                            Map<Term, Optional<Model>> outputExtensions) implements RuleApplication {

        @Override
        public String toString() {
            return getString(0, "");
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof ExistsElimRuleApplication(Exists otherInput, var _)) {
                return this.input.equals(otherInput);
            }
            return false;
        }

        @Override
        public int hashCode() {
            return input.hashCode();
        }

        @Override
        public String getString(int indentation, String delim) {
            if (indentation == 0) {
                StringBuilder sb = new StringBuilder("∃Elim: ");
                sb.append(input).append(" ->");
                if (!outputExtensions.isEmpty())
                    sb.append(" \n");
                for (var entry : outputExtensions.entrySet()) {
                    sb.append("  ").append(entry.getKey()).append(": ");
                    entry.getValue().ifPresent(m -> sb.append(m.createString(2, delim)).append("\n"));
                }
                return sb.toString();
            }
            String outerIndent = "  ".repeat(Math.max(0, indentation)) + delim + " ";
            StringBuilder sb = new StringBuilder(outerIndent)
                    .append("∃Elim : ")
                    .append(input);
            if (!outputExtensions.isEmpty())
                sb.append(" : \n");
            for (var entry : outputExtensions.entrySet()) {
                sb.append("  ".repeat(indentation + 1)).append(delim).append(" ").append(entry.getKey()).append(": ");
                entry.getValue().ifPresent(m -> sb.append(m.createString(indentation + 2, delim.repeat(2))).append("\n"));
            }
            return sb.toString();
        }
    }
}
