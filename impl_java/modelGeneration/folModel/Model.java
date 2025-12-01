package modelGeneration.folModel;

import fol.Substitution;
import fol.Unifier;
import fol.formula.Equals;
import fol.formula.Formula;
import fol.formula.Not;
import fol.term.Term;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import modelGeneration.Salient;
import modelGeneration.axioms.ConsistencyCheck;
import modelGeneration.folModel.rules.RuleApplication;
import modelGeneration.ontModel.invokations.InvokedSituation;

public class Model {
    private final Set<Formula> formulas;
    private final Set<Salient<Term>> individuals;
    private final Map<RuleApplication, Set<Model>> extensions;
    private final Set<RuleApplication> ruleApplications;
    private Substitution substitution;
    private boolean complete = false;
    private Model parent;
    private int sentenceDepth = 0;
    public final long id;
    private static final AtomicLong NEXT_ID = new AtomicLong();
    // Ontology resolution
    private final Set<Salient<InvokedSituation>> invokedSituations;

    public Substitution getSubstitution() {
        return substitution.copy();
    }

    public void setIndividualsSalience(Set<Salient<Term>> newSalience) {
        for (var ind : newSalience) {
            individuals.remove(ind);
            individuals.add(ind);
        }
    }

    public void addInvokedSituations(List<Salient<InvokedSituation>> invokedSituations) {
        this.invokedSituations.addAll(invokedSituations);
    }

    public int getSentenceDepth() {
        return sentenceDepth;
    }

    public void setSentenceDepth(int sentenceDepth) {
        this.sentenceDepth = sentenceDepth;
    }

    public boolean isComplete() {
        return complete;
    }

    public void complete() {
        this.complete = true;
    }

    public void setParent(Model parent) {
        this.parent = parent;
    }

    public Model(Set<Formula> formulas, int sentenceDepth) {
        this.formulas = new HashSet<>(formulas);
        this.individuals = new HashSet<>();
        this.ruleApplications = new HashSet<>();
        this.extensions = new HashMap<>();
        this.substitution = new Substitution();
        this.invokedSituations = new HashSet<>();
        this.sentenceDepth = sentenceDepth;
        this.id = NEXT_ID.getAndIncrement(); // Monotonic identifier for queue ordering
    }

    public Model(Set<Formula> formulas, int sentenceDepth, Model parent) {
        this(formulas, sentenceDepth);
        this.parent = parent;
    }

    public Model(Set<Formula> formulas, int sentenceDepth, Set<Salient<Term>> individuals) {
        this(formulas, sentenceDepth);
        this.individuals.addAll(individuals.stream().map(Salient::copy).toList());
    }

    public Model(Set<Formula> formulas, int sentenceDepth, Model parent, Set<Salient<Term>> individuals) {
        this(formulas, sentenceDepth, parent);
        this.individuals.addAll(individuals);
    }

    public Model(Set<Formula> formulas, int sentenceDepth, Set<Salient<Term>> individuals,
            Map<RuleApplication, Set<Model>> extensions, Set<RuleApplication> ruleApplications, Model parent) {
        this(formulas, sentenceDepth);
        this.individuals.addAll(individuals);
        this.extensions.putAll(extensions);
        this.ruleApplications.addAll(ruleApplications);
        this.parent = parent;
    }

    public Model copy() {
        var newModel = new Model(formulas, sentenceDepth, individuals, extensions, ruleApplications, parent);
        newModel.substitution = substitution.copy();
        newModel.complete = false;
        newModel.invokedSituations.addAll(invokedSituations.stream().map(Salient::copy).toList());
        return newModel;
    }

    public boolean checkConsistency(Set<ConsistencyCheck> consistencyChecks) {
        for (final var formula : formulas) {
            if (formula instanceof Not(Formula innerFormula)) {
                // !a; a
                if (formulas.contains(innerFormula))
                    return false;
                if (innerFormula instanceof Equals eqFormula) {
                    // c != c
                    if (eqFormula.left().equals(eqFormula.right()))
                        return false;
                    // c != d ; c = d
                    if (formulas.contains(eqFormula))
                        return false;
                }
            }
            if (formula instanceof Equals eqFormula) {
                // unify and update substitution, otherwise fail
                final var maybeSubstitution = Unifier.unify(eqFormula.left(), eqFormula.right(), substitution);
                if (maybeSubstitution.isEmpty())
                    return false;
            }
        }
        for (final var consistencyCheck : consistencyChecks) {
            if (!consistencyCheck.check(this)) {
                return false;
            }
        }
        return true;
    }

    public void applySubstitution(Substitution substitution) {
        var fCopy = List.copyOf(formulas);
        for (var formula : fCopy) {
            formulas.add(formula.applySub(substitution));
        }
        this.substitution = this.substitution.compose(substitution);
    }

    public Set<RuleApplication> getRuleApplications() {
        return new HashSet<>(ruleApplications);
    }

    public void addRuleApplications(Set<RuleApplication> ruleApplications) {
        this.ruleApplications.addAll(ruleApplications);
    }

    public Set<Formula> getFormulas() {
        return formulas;
    }

    public Model clip() {
        
        Model newModel = copy();
        newModel.complete = complete;
        newModel.parent = null;
        return newModel;
    }

    public void addFormulas(Set<Formula> formulas) {
        this.formulas.addAll(formulas);
        for (var formula : formulas) {
            this.formulas.add(formula.applySub(substitution));
        }
    }

    public Set<Salient<Term>> getIndividuals() {
        return individuals;
    }

    public void addSalientIndividuals(Set<Salient<Term>> individuals) {
        this.individuals.addAll(individuals);
    }

    public void addIndividuals(Set<Term> individuals, double salience) {
        individuals.forEach(ind -> this.individuals.add(new Salient<>(ind, salience)));
    }

    public void addIndividuals(Set<Term> individuals) {
        addIndividuals(individuals, Salient.FULL);
    }

    public Model getParent() {
        return parent;
    }

    public Map<RuleApplication, Set<Model>> getExtensions() {
        return Map.copyOf(extensions);
    }

    public void addExtensions(Map<RuleApplication, Set<Model>> extensions) {
        this.extensions.putAll(extensions);
    }

    public String createString(int indentation, String delim) {
        StringBuilder sb = new StringBuilder();
        sb.append("Model (%d):\n".formatted(invokedSituations.size()));
        String outerIndent = "  ".repeat(Math.max(0, indentation)) + delim + " ";
        String nestedIndent = "  ".repeat(Math.max(0, indentation + 1)) + delim + " ";
        if (!formulas.isEmpty()) {
            var formulasStrings = formulas.stream()
                    .sorted((Formula f1, Formula f2) -> {
                        int cmp = Integer.compare(f1.countLiterals(), f2.countLiterals());
                        if (cmp != 0)
                            return cmp;
                        return f1.toString().compareTo(f2.toString());
                    })
                    .map(Object::toString)
                    .collect(Collectors.toList());
            sb.append(outerIndent)
                    .append("Formulas:\n")
                    .append(nestedIndent)
                    .append(String.join("\n  * ", formulasStrings))
                    .append("\n");
        }
        if (!individuals.isEmpty())
            sb.append(outerIndent)
                    .append("Individuals:\n")
                    .append(nestedIndent)
                    .append(
                            String.join("\n  * ", individuals.stream()
                                    .sorted(Comparator.reverseOrder())
                                    .map(Object::toString).toList()))
                    .append("\n");
        if (!ruleApplications.isEmpty())
            sb.append(outerIndent)
                    .append("Rule applications:\n")
                    .append(
                            String.join("\n", ruleApplications.stream()
                                    .map(ra -> ra.getString(indentation + 1, delim)).toList()))
                    .append("\n");
        return sb.toString();
    }

    public String createString(int indentation) {
        return createString(indentation, "*");
    }

    @Override
    public String toString() {
        return createString(0);
    }
}
