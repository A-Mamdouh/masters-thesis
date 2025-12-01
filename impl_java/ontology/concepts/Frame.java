package ontology.concepts;

import fol.Language;
import fol.formula.And;
import fol.formula.Exists;
import fol.formula.Formula;
import fol.term.Constant;
import fol.term.Term;
import fol.term.Variable;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import modelGeneration.Salient;
import modelGeneration.folModel.Model;

public record Frame(
        String name,
        List<Role> roles
) {
    public Model getModel(Variable situationVar) {
        // \exists_f:frame(sVar, f, name).\bigWedge[roles]
        final var frameVar = Variable.make();
        final Set<Model> rolesModels = roles.stream()
                .map(r -> r.getModel(frameVar))
                .collect(Collectors.toSet());

        Formula rolesConj = null;
        final Model frameModel = new Model(Set.of(), 1);
        for (var model : rolesModels) {
            for (var form : model.getFormulas()) {
                if (rolesConj == null) {
                    rolesConj = form;
                } else {
                    rolesConj = new And(rolesConj, form);
                }
            }
            frameModel.addSalientIndividuals(model.getIndividuals());
        }
        frameModel.addSalientIndividuals(Set.<Salient<Term>>of(new Salient<>(new Constant(name), Salient.FULL)));
        final var precondition = Language.Predicates.frame(situationVar, frameVar, new Constant(name));
        final var frameFormula = new Exists(frameVar, precondition, rolesConj);
        frameModel.addFormulas(Set.of(frameFormula));
        return frameModel;
    }
}