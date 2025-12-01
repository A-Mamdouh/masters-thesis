package ontology.concepts;

import fol.Language;
import fol.formula.Exists;
import fol.term.Constant;
import fol.term.Term;
import fol.term.Variable;
import java.util.Set;
import modelGeneration.Salient;
import modelGeneration.folModel.Model;

public record Role(String name, String semType) {
    public Model getModel(Variable frameVar) {
        // \exists_r:semType(r, semType).role(fVar, r, name)
        final var roleVar = Variable.make();
        final var innerForm = Language.Predicates.role(frameVar, roleVar, new Constant(name));
        final var precondition = Language.Predicates.semType(roleVar, new Constant(semType));
        var formula = new Exists(roleVar, precondition, innerForm);
        Model model = new Model(Set.of(), 1);
        final Set<Salient<Term>> individuals = Set.<Salient<Term>>of(
                new Salient<>(new Constant(name), Salient.FULL),
                new Salient<>(new Constant(semType), Salient.FULL)
        );
        model.addSalientIndividuals(individuals);
        model.addFormulas(Set.of(formula));
        return model;
    }
}