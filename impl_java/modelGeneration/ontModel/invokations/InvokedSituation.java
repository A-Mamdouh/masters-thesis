package modelGeneration.ontModel.invokations;

import fol.Language;
import fol.formula.*;
import fol.term.Constant;
import fol.term.Term;
import fol.term.Variable;
import ontology.ParsedSentence;
import ontology.concepts.Situation;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class InvokedSituation {
    private final Situation situation;
    private final ParsedSentence sentence;
    private final InvokedFrame invokedFrame;

    public InvokedSituation(Situation situation, ParsedSentence sentence, InvokedFrame invokedFrame) {
        this.situation = situation;
        this.sentence = sentence;
        this.invokedFrame = invokedFrame;
    }

    /**
     * Try to invoke the situation by finding a frame that matches the sentence
     * @param sentence
     * @return invoked situation given the sentence
     */
    public static Set<InvokedSituation> invokeSituationFromSentence(ParsedSentence sentence, Situation situation) {
        if(sentence.isNegated()) {
            for(final var frame : situation.frames()) {
                final var maybeInvoked = InvokedFrame.invokeNegated(frame, sentence);
                if(!maybeInvoked.isEmpty()) {
                    return maybeInvoked.stream()
                            .map(iFrame -> new InvokedSituation(situation, sentence, iFrame))
                            .collect(Collectors.toSet());
                }
            }
            return Set.of();
        }
        for(final var frame : situation.frames()) {
            final var maybeInvoked = InvokedFrame.invoke(frame, sentence);
            if(maybeInvoked.isPresent()) {
                return Set.of(new InvokedSituation(situation, sentence, maybeInvoked.get()));
            }
        }
        return Set.of();
    }

    public Set<Formula> getFormulas() {
        // First, get formulas for the semantic types of known individuals
        final Set<Formula> formulas = new HashSet<>();
        for(final var ind : sentence.individuals()) {
            if(ind.isKnown()) {
                formulas.add(Language.Predicates.semType(ind.name(), ind.semType()));
            }
        }
        // Add the formulas of the frames and roles with the individuals filled in
        // \exists[s] : situation(s, name) . frame1 ^ frame2 ^ ...
        final var situationVar = Variable.make();
        final var precondition = Language.Predicates.situation(new Constant(situation.name()), situationVar);
        formulas.add(new Exists(situationVar, precondition, invokedFrame.getFormula(situationVar)));
        return formulas;
    }

    public Set<Term> getIndividualsAsTerms() {
        // Returns the set of individuals matched in this situation. Unknown individuals are skipped
        return sentence.individuals().stream()
                .filter(ParsedSentence.TypedIndividual::isKnown)
                .map(i -> new Constant(i.name()))
                .collect(Collectors.toSet());
    }

    public InvokedFrame getInvokedFrame() {
        return invokedFrame;
    }

}
