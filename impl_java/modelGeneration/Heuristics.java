package modelGeneration;

import java.util.Comparator;
import modelGeneration.folModel.Model;

public class Heuristics {

    public interface Heuristic extends Comparator<Model> {
        public String getName();

        public int getDrainSize();
    }

    public static class BFS implements Heuristic {
        @Override
        public int compare(Model model, Model t1) {
            return Long.compare(model.id, t1.id);
        }

        @Override
        public String getName() {
            return "BFS";
        }

        @Override
        public int getDrainSize() {
            return -1;
        }
    }

    public static class DFS implements Heuristic {
        @Override
        public int compare(Model model, Model t1) {
            return Long.compare(t1.id, model.id);
        }

        @Override
        public String getName() {
            return "DFS";
        }

        @Override
        public int getDrainSize() {
            return 1;
        }
    }

    public static class CustomHeuristic implements Heuristic {
        @Override
        public int compare(Model model, Model t1) {
            // Prefer deeper models (conversationally)
            int compSize = Integer.compare(model.getSentenceDepth(), t1.getSentenceDepth());
            if (compSize != 0)
                return compSize;
            // Then smallest assumptions about individuals
            compSize = Integer.compare(model.getIndividuals().size(), t1.getIndividuals().size());
            if (compSize != 0)
                return compSize;
            // Lastly, smallest number of formulas
            compSize = Integer.compare(t1.getFormulas().size(), model.getFormulas().size());
            return compSize;
        }

        @Override
        public String getName() {
            return "Custom Heuristic";
        }

        @Override
        public int getDrainSize() {
            return 12;
        }
    }
}
