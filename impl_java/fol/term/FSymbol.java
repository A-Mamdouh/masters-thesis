package fol.term;

public record FSymbol(String name, int arity) {
    public String toString() {
        return name + "\\" + arity;
    }
}
