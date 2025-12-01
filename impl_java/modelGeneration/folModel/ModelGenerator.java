package modelGeneration.folModel;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.stream.Collectors;
import modelGeneration.Salient;
import modelGeneration.axioms.ConsistencyCheck;
import modelGeneration.folModel.rules.*;
import modelGeneration.ontModel.invokations.InvokedSituation;
import ontology.ParsedSentence;
import ontology.concepts.Situation;

public class ModelGenerator {
    private final PriorityBlockingQueue<Model> searchTree;
    private final List<InferenceRule> rulesBranching;
    private final List<InferenceRule> rulesNonBranching;
    private final List<Model> results;
    private final ExecutorService executor;
    private final int poolSize;
    private final List<ParsedSentence> sentences;
    private final Set<ConsistencyCheck> consistencyChecks ;
    private int timeoutms = 0;
    private int numExploredModels = 0;
    private final int drainSize;
    private int closedModels = 0;

    public ModelGenerator(Model intialModel, Comparator<Model> heuristic, List<InferenceRule> inferenceRules, Set<ConsistencyCheck> consistencyChecks, int poolSize, int drainSize) {
        rulesBranching = inferenceRules.stream().filter(InferenceRule::isBranching).toList();
        rulesNonBranching = inferenceRules.stream().filter(rule -> !rule.isBranching()).toList();
        results = new LinkedList<>();
        this.poolSize = Math.max(1, poolSize);
        searchTree = new PriorityBlockingQueue<>(this.poolSize, heuristic);
        executor = Executors.newFixedThreadPool(this.poolSize);
        searchTree.add(intialModel);
        this.sentences = new LinkedList<>();
        this.consistencyChecks = consistencyChecks;
        if(drainSize == -1) {
            this.drainSize = this.poolSize;
        } else {
            this.drainSize = drainSize;
        }
    }

    public ModelGenerator(Model intialModel, Comparator<Model> heuristic, Set<ConsistencyCheck> consistencyChecks, int poolSize, int drainSize) {
        this(intialModel, heuristic, getDefaultInferenceRules(), consistencyChecks, poolSize, drainSize);
    }

    private static List<InferenceRule> getDefaultInferenceRules() {
        return List.of(new AndElim(), new ExistElim(), new ForallElim(), new NotElim(), new OrElim(), new EqElim());
    }

    public void setTimeout(int timeoutms) {
        this.timeoutms = timeoutms;
    }


    /**
     * Match the sentence to frames on a situation level and return an *ordered* list from best to worst match
     * TODO: Add ordering
     * @param sentence
     * @return an *ordered* list from best to worst matching situation
     */
    private List<InvokedSituation> matchSentenceToSituations(ParsedSentence sentence, Set<Situation> ontology) {
        List<InvokedSituation> matches = new ArrayList<>();
        for(final var situation : ontology) {
            matches.addAll(InvokedSituation.invokeSituationFromSentence(sentence, situation));
        }
        return matches;
    }

    public void addParsedSentence(ParsedSentence sentence, Set<Situation> ontology) {
        var matches = matchSentenceToSituations(sentence, ontology);
        var leaves = new LinkedList<>(results);
        leaves.addAll(searchTree);
        searchTree.clear();
        results.clear();
        // For every model in the queue and results:
        //  - For every match for the sentence:
        //      - add the match to the model
        //      - add the extended model back to the queue
        for(var model : leaves) {
            for(var invokedSituation : matches) {
                var newModel = model.copy();
                newModel.addInvokedSituations(List.of(new Salient<>(invokedSituation, Salient.FULL)));
                // Add formulas from the invoked situation into the new model
                for(final var formula : invokedSituation.getFormulas()) {
                    newModel.addFormulas(Set.of(formula));
                }
                // Decrease the salience of existing model individuals
                final var updatedSalience = newModel.getIndividuals().stream()
                        .map(ind -> new Salient<>(ind.obj(), ind.salience() * Salient.DECAY_RATE))
                        .collect(Collectors.toSet());
                newModel.setIndividualsSalience(updatedSalience);
                // Add new individuals to the model
                for(final var ind : invokedSituation.getIndividualsAsTerms()) {
                    newModel.addIndividuals(Set.of(ind));
                }
                newModel.setSentenceDepth(this.sentences.size() + 1);
                searchTree.add(newModel);
            }
        }
        this.sentences.add(sentence);
    }

    public void exit() {
        executor.shutdown();
    }

    public Optional<Model> generateModel(boolean verbose) {
        long startTime = System.currentTimeMillis();
        boolean timedOut = false;
        while(results.size() < 5 && !searchTree.isEmpty()) {
            if(timeoutms > 0 && System.currentTimeMillis() - startTime > timeoutms) {
                timedOut = true;
                break;
            }
            List<Model> models = new ArrayList<>(drainSize);
            searchTree.drainTo(models, drainSize);
            numExploredModels += models.size();
            if(models.isEmpty()) break;
            var futures = new ArrayList<CompletableFuture<WorkerResult>>(models.size());
            for (int idx = 0; idx < models.size(); idx++) {
                final int index = idx;
                final var model = models.get(idx);
                final var checksCopy = consistencyChecks.stream().map(e -> e.getCopy()).collect(Collectors.toSet());
                futures.add(CompletableFuture.supplyAsync(() -> processModel(index, model, checksCopy), executor));
            }
            WorkerResult[] batchResults = new WorkerResult[models.size()];
            for (var future : futures) {
                try {
                    var workerResult = future.get();
                    if (workerResult != null && workerResult.index() < batchResults.length) {
                        batchResults[workerResult.index()] = workerResult;
                    }
                } catch (Exception e) {
                    System.err.println("Error while generating a model");
                    System.err.println(e.getMessage());
                }
            }
            for (WorkerResult workerResult : batchResults) {
                if (workerResult == null) continue;
                for (var newModel : workerResult.newModels()) {
                    searchTree.add(newModel.clip());
                }
                workerResult.result().ifPresent(result -> {
                    results.add(result.clip());
                    closedModels++;
                });
            }
        }
        if(verbose) {
            System.out.println("Timed out: " + timedOut);
        }
        if(!results.isEmpty()) {
            return Optional.of(results.getFirst());
        }
        for(final var model : searchTree) {
            if(model.checkConsistency(consistencyChecks)) {
                return Optional.of(model);
            }
        }
        return Optional.empty();
    }

    public void clearTopResult() {
        if(results.isEmpty()) return;
        results.removeFirst();
    }

    public int getNumExploredModels() {
        return numExploredModels;
    }

    public int getClosedModelCount() {
        return closedModels;
    }

    public int getOpenModelCount() {
        return searchTree.size();
    }

    private WorkerResult processModel(int index, Model currentModel, Set<ConsistencyCheck> consistencyChecks) {
        final List<Model> newFrontier = new LinkedList<>();
        if(currentModel.isComplete()){
            return new WorkerResult(index, Optional.of(currentModel), newFrontier);
        }
        boolean modelChanged = false;
        for (boolean done = false; !done; ) {
            done = true;
            for(var rule : rulesNonBranching) {
                if(!rule.apply(currentModel)) continue;
                modelChanged = true;
                done = false;
            }
        }
        boolean branched = false;
        for(var rule : rulesBranching) {
            if(!rule.apply(currentModel)) continue;
            branched = true;
            break;
        }

        if(!currentModel.checkConsistency(consistencyChecks)) {
            return new WorkerResult(index, Optional.empty(), List.of());
        }

        if(branched) {
            for(var branchSet : currentModel.getExtensions().values()) {
                newFrontier.addAll(branchSet);
            }
        } else {
            if(!modelChanged)
                currentModel.complete();
            newFrontier.add(currentModel);
        }
        return new WorkerResult(index, Optional.empty(), newFrontier);
    }

    private record WorkerResult(int index, Optional<Model> result, List<Model> newModels) {}

}
