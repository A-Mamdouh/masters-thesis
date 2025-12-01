package ontology.concepts;

import fol.Language;
import fol.formula.Exists;
import fol.term.Constant;
import fol.term.Variable;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import modelGeneration.folModel.Model;

public record Situation(
        String name,
        List<Frame> frames
) {
    public Model getModel() {
        // \exists_s:situation(s, name).frame1 ; \exists_s:situation(s, name).frame2 ; ....
        final var situationVar = Variable.make();
        final var situationPrecondition = Language.Predicates.situation(situationVar, new Constant(name));
        final var frameModels = frames.stream().map(f -> f.getModel(situationVar)).collect(Collectors.toSet());
        final Model outputModel = new Model(Set.of(), 1);
        for (final var frameModel : frameModels) {
            outputModel.addSalientIndividuals(frameModel.getIndividuals());
            for (var frameFormula : frameModel.getFormulas()) {
                outputModel.addFormulas(Set.of(new Exists(situationVar, situationPrecondition, frameFormula)));
            }
        }
        return outputModel;
    }
}