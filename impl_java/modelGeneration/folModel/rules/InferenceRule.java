package modelGeneration.folModel.rules;

import modelGeneration.folModel.Model;

public interface InferenceRule {

    /**
     * Applied the rule assuming that it is applicable. Extending, or expanding it according to the rule
     * Exceptions are thrown if the rule is not applicable.
     * This method mutates the input model
     *
     * @param model the model to apply the rule to
     */
    boolean apply(final Model model);

    boolean isBranching();
}
