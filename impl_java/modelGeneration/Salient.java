package modelGeneration;


public record Salient<T>(T obj, Double salience) implements Comparable<Salient<T>> {

    public static final double FULL = 1.0;
    public static final double DECAY_RATE = 0.9;

    @Override
    public int compareTo(Salient<T> tSalient) {
        return salience.compareTo(tSalient.salience);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Salient s && s.obj.equals(obj);
    }


    public Salient<T> copy() {
        return new Salient<>(obj, salience);
    }

    @Override
    public int hashCode() {
        return obj.hashCode();
    }

}
