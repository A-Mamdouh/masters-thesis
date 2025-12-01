package ontology;

import java.util.List;

public class ParsedSentence {

    private final String verb;
    private final List<TypedIndividual> individuals;
    private final boolean isNegated;
    private final String originalSentence;

    public ParsedSentence(String verb, List<TypedIndividual> individuals, boolean isNegated) {
        this(verb, individuals, isNegated, null);
    }

    public ParsedSentence(String verb, List<TypedIndividual> individuals, boolean isNegated, String originalSentence) {
        assert verb != null;
        this.verb = verb;
        this.individuals = individuals;
        this.isNegated = isNegated;
        this.originalSentence = originalSentence;
    }

    public String getOriginalString() {
        return originalSentence;
    }

    public String toString() {
        return String.format("ParsedSentence[verb=%s, individuals=%s, isNegated=%b, originalSentence=%s]", verb,
                individuals.stream().map(Object::toString).reduce((a, b) -> a + ", " + b).orElse(""),
                isNegated, originalSentence);
    }

    public ParsedSentence(String verb, List<TypedIndividual> individuals) {
        this(verb, individuals, false);
    }

    public boolean isNegated() {
        return isNegated;
    }

    public List<TypedIndividual> individuals() {
        return List.copyOf(individuals);
    }

    public String verb() {
        return verb;
    }

    public static TypedIndividual createUnknownIndividual(String semType) {
        return new TypedIndividual(UNKNOWN_INDIVIDUAL, semType);
    }

    private static final String UNKNOWN_INDIVIDUAL = "UNKOWN_IND";

    public record TypedIndividual(String name, String semType) {
        @Override
        public String toString() {
            return name + ":" + semType;
        }

        public boolean isKnown() {
            return !name.equals(UNKNOWN_INDIVIDUAL);
        }
    }
}
