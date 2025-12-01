package fol.formula;

public record PSymbol(String name, int arity) {
    public String toString() {
        return name + "\\" + arity;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof PSymbol other)) return false;
        return name.equals(other.name) && arity == other.arity;
    }

    @Override
    public int hashCode() {
        return (name + "\\" + arity).hashCode();
    }
}
