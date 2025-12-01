import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.*;
import modelGeneration.Heuristics;
import modelGeneration.axioms.ConsistencyCheck;
import modelGeneration.axioms.FramesLogicCheck;
import modelGeneration.axioms.MurderChecks;
import modelGeneration.folModel.*;
import ontology.Data.Situations;
import ontology.ParsedSentence;
import ontology.Stories;
import ontology.concepts.Situation;

public class Main {

    public static void main(String[] args) {

        int poolSize = 12;
        if(args.length > 0) {
            try {
                poolSize = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("First argument must be an integer representing the pool size. Using default of 12.");
            }
        }

        int timeOut = 200; // ms
        if(args.length > 1) {
            try {
                timeOut = Integer.parseInt(args[1]);
            } catch (NumberFormatException e) {
                System.err.println("Second argument must be an integer representing the timeout in ms. Using default of 200ms.");
            }
        }

        try {
            final var outputFileName = String.format("test_results_%d_workers_%dms.txt", poolSize, timeOut);
            System.setOut(new PrintStream(new FileOutputStream(outputFileName), true));
        } catch (FileNotFoundException e) {
            System.err.println("Could not redirect output to file, printing to console instead.");
            e.printStackTrace();
        }

        final List<Heuristics.Heuristic> heuristics = List.of(
                new Heuristics.DFS(),
                new Heuristics.BFS()
        );
        final Heuristics.Heuristic baseHeuristic = new Heuristics.CustomHeuristic();
        for (var story : Stories.stories) {
            System.out.println("Results for story:\n\n  " + String.join("\n  ", story.stream().map(ParsedSentence::getOriginalString).toList()) + "\n");
            compareHeuristics(baseHeuristic, heuristics, story, poolSize, timeOut);
            System.out.println("---------------------------------------------------");
            System.out.println("---------------------------------------------------\n");
        }
    }

    public record HeuristicStatistics(Heuristics.Heuristic heuristic, long timeMs, int numExploredModels,
            int closedModels, int openModels, Optional<Model> model) {
        @Override
        public String toString() {
            return String.format(
                    "HeuristicStatistics[heuristic=%s, timeMs=%d, exploredModels=%d, closedModels=%d, openModels=%d, modelFound=%b]",
                    heuristic.getName(), timeMs, numExploredModels, closedModels, openModels, !model.isEmpty());
        }
    }

    private static List<HeuristicStatistics> compareHeuristics(Heuristics.Heuristic baseHeuristic,
            List<Heuristics.Heuristic> otherHeuristics, List<ParsedSentence> dialogue, int poolSize, int timeOut) {

        Set<Situation> ontology = Set.of(
                Situations.celebration, Situations.murder);

        // Create custom consistency checks
        final Set<ConsistencyCheck> consistencyChecks = Set.of(
                new MurderChecks(),
                new FramesLogicCheck());

        List<HeuristicStatistics> output = new LinkedList<>();

        final var baseResult = testHeuristic(baseHeuristic, consistencyChecks, poolSize, timeOut, ontology,
                new LinkedList<>(dialogue));
        if (baseResult.model().isEmpty()) {
            throw new RuntimeException("Base heuristic does not reach a model");
        }
        final var baseModel = baseResult.model().get();
        output.add(baseResult);
        System.out.println(baseResult.toString());
        for (final var heuristic : otherHeuristics) {
            final var result = testHeuristicWithTargetHeuristic(heuristic, consistencyChecks, poolSize, timeOut,
                    ontology,
                    new LinkedList<>(dialogue), baseModel, baseHeuristic);
            output.add(result);
            System.out.println(result.toString());
        }
        // Create the model generator
        return output;
    }

    private static HeuristicStatistics testHeuristic(Heuristics.Heuristic heuristic,
            Set<ConsistencyCheck> consistencyChecks,
            int poolSize, int timeOut, Set<Situation> ontology, Queue<ParsedSentence> sentences) {

        final var initialModel = new Model(Set.of(), 0);
        ModelGenerator modelGenerator = new ModelGenerator(initialModel, heuristic, consistencyChecks, poolSize, heuristic.getDrainSize());
        modelGenerator.setTimeout(timeOut);

        Model generatedModel = null;
        final var startTime = System.currentTimeMillis();
        while (!sentences.isEmpty()) {

            modelGenerator.addParsedSentence(sentences.poll(), ontology);
            var maybeModel = modelGenerator.generateModel(false);
            if (maybeModel.isEmpty()) {
                generatedModel = null;
                break;
            }
            generatedModel = maybeModel.get();
        }
        final var endTime = System.currentTimeMillis();
        modelGenerator.exit();
        return new HeuristicStatistics(
                heuristic,
                endTime - startTime,
                modelGenerator.getNumExploredModels(),
                modelGenerator.getClosedModelCount(),
                modelGenerator.getOpenModelCount(),
                Optional.ofNullable(generatedModel));
    }

    private static HeuristicStatistics testHeuristicWithTargetHeuristic(Heuristics.Heuristic heuristic,
            Set<ConsistencyCheck> consistencyChecks,
            int poolSize, int timeOut, Set<Situation> ontology, Queue<ParsedSentence> sentences, Model targetModel,
            Heuristics.Heuristic baseHeuristic) {

        final var initialModel = new Model(Set.of(), 0);
        ModelGenerator modelGenerator = new ModelGenerator(initialModel, heuristic, consistencyChecks, poolSize, heuristic.getDrainSize());
        modelGenerator.setTimeout(timeOut);

        Model generatedModel = null;
        final var startTime = System.currentTimeMillis();
        long comparisonTime = 0;
        while (!sentences.isEmpty()) {

            modelGenerator.addParsedSentence(sentences.poll(), ontology);
            final var maybeModel = modelGenerator.generateModel(false);
            if (maybeModel.isEmpty()) {
                generatedModel = null;
                break;
            }
            generatedModel = maybeModel.get();
        }
        if (generatedModel != null) {
            while (true) {
                final var compStart = System.currentTimeMillis();
                final var equivalent = baseHeuristic.compare(generatedModel, targetModel) <= 0;
                comparisonTime += System.currentTimeMillis() - compStart;
                if (equivalent) {
                    break;
                }
                modelGenerator.clearTopResult();
                final var maybeModel = modelGenerator.generateModel(false);
                if (maybeModel.isEmpty()) {
                    generatedModel = null;
                    break;
                }
                generatedModel = maybeModel.get();
            }
        }
        final var endTime = System.currentTimeMillis();
        modelGenerator.exit();
        return new HeuristicStatistics(
                heuristic,
                endTime - startTime - comparisonTime,
                modelGenerator.getNumExploredModels(),
                modelGenerator.getClosedModelCount(),
                modelGenerator.getOpenModelCount(),
                Optional.ofNullable(generatedModel));
    }
}
