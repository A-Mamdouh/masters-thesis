package modelGeneration.folModel.rules;

public interface RuleApplication {
    @Override
    boolean equals(Object obj);

    String getString(int indentation, String delim);
}
