## Model generation

The Python port now exposes the model generation strategies from the Java design under
`nls_python.model_generation`:

- `PriorityModelGenerator` mirrors the priority-queue search from the Java implementation and
  evaluates tableau nodes with a configurable heuristic/cost function.
- `BranchQueueModelGenerator` assigns chunks of open branches to worker-local queues. Each thread
  keeps expanding its queue until every branch either closes or produces new open leaves, which are
  then shared across workers.

Both generators accept the same inputs:

```python
from nls_python.model_generation import (
    PriorityModelGenerator,
    depth_biased_cost,
    TableauModel,
)

initial_model = TableauModel(formulas={...})
generator = PriorityModelGenerator(
    initial_model=initial_model,
    cost_function=depth_biased_cost,
    consistency_checks=(),
)
model = generator.generate_model()
```

The `heuristics` module provides a `ModelCostFunction` protocol, a `StaticCostFunction` helper for
deterministic heuristics, and a `NeuralPolicyAdapter` that can later call into neural scoring
functions. Feature extraction is handled by `basic_model_features`, so swapping heuristics or adding
learned policies only requires providing a callable that consumes those features.

## Policy helpers

`nls_python.model_generation.policies` exposes ready-to-use search policies mirroring the Java
examples (`BFS`, `DFS`, and the custom conversational heuristic). Policies bundle both the scoring
function and the appropriate `drain_size`, making it easy to configure generators:

```python
from nls_python.model_generation import PriorityModelGenerator, BFS_POLICY

generator = PriorityModelGenerator(
    initial_model=model,
    cost_function=BFS_POLICY.cost_function,
    drain_size=BFS_POLICY.drain_size,
)
```

To plug in a learned scorer, call `make_neural_policy` with any predictor operating on
`ModelFeatures`. The helper returns a `SearchPolicy` that can be registered and reused alongside the
hand-authored heuristics.

## Story experiments

`python -m nls_python.model_generation.story_experiments` reproduces the Java `Main` experiments in
Python. The script iterates over the bundled stories, runs the base policy (CustomJava), compares
additional policies (DFS/BFS by default), and writes a log similar to the original
`test_results_{workers}_{timeout}.txt` files. Example:

```bash
uv run python -m nls_python.model_generation.story_experiments --pool-size 8 --timeout-ms 200
```

Use `--policies` to compare additional registered policies or `--base-policy` to swap the baseline
heuristic. Output is written both to stdout and to `test_results_{pool}_workers_{timeout}ms.txt`
unless `--output` overrides the path.

## Annotated stories

Example stories used throughout the thesis now live in the annotated JSON files under `data/`
(`train.json`, `val.json`, `test.json`). Each entry captures the raw sentence text, roles, semantic
types, and anaphora links needed by the tableau builder. Load them with
`nls_python.model_generation.story.dataset.load_annotated_stories([Path("data/test.json")])` and feed
the resulting `StoryExample` objects directly into the generators or policy trainers.

## Neural workflow

The neural heuristics follow a three-step workflow inspired by the thesis experiments:

1. **Dataset generation** – Replay stories, capture tableau features/narratives, and (optionally) emit
   train/validation splits:

   ```bash
   uv run python -m nls_python.model_generation.training_data \
     --output data/all.jsonl \
     --train-output data/train.jsonl \
     --val-output data/val.jsonl \
     --policies CustomJava \
     --datasets data/train.json data/val.json data/test.json
   ```

   Records include rank metadata, rule counts, and the sentence-level narrative used to train neural
   policies.

2. **Training** – Fit the `NarrativeNeuralNetwork` on any JSONL dataset produced above, logging loss
   and accuracy curves to a directory for inspection:

   ```bash
   uv run python -m nls_python.model_generation.neural_training \
     --dataset data/train.jsonl \
     --output checkpoints/neural.pt \
     --epochs 10 --lr 3e-4 --device cpu \
     --log-dir logs/train
   ```

3. **Evaluation & experiments** – Load the checkpoint into the story experiment harness or the
   evaluation helper (which can also write per-story correlation plots):

   ```bash
   # Compare against DFS/BFS/CustomJava in the full experiment suite
   uv run python -m nls_python.model_generation.story_experiments \
     --neural-checkpoint checkpoints/neural.pt --policies DFS BFS NeuralPolicy

   # Print Spearman correlations + model-found rate across the annotated sets
   uv run python -m nls_python.model_generation.neural_evaluation \
     --neural-checkpoint checkpoints/neural.pt \
     --datasets data/test.json \
     --top-k 5 \
     --plot-dir logs/eval
```

This loop (dataset → training → experiments) mirrors the thesis workflow, making it easy to retrain
heuristics or plug in alternative ranking sources for supervision.

### Symbolic policy distillation

To distill every hand-authored heuristic into its own neural mimic (and capture top-k metrics automatically),
use the batch runner:

```bash
uv run python -m nls_python.model_generation.policy_experiments \
  --policies CustomJava DepthBiased BFS DFS \
  --datasets data/train.json data/val.json data/test.json \
  --output-dir experiments/policies \
  --epochs 5 --lr 3e-4 --top-k 5 \
  --summary experiments/policies/summary.json
```

Each policy receives its own dataset (`experiments/policies/<policy>.jsonl`), checkpoint, log directory, and a
summary entry containing model-found rate, average Spearman, and the requested top-k accuracy. Rerun with
`--overwrite` if you need to regenerate datasets or checkpoints.

## Automated training pipeline

For a single command that splits a dataset, trains a neural policy, and evaluates it, use the pipeline CLI:

```bash
uv run python -m nls_python.model_generation.training_pipeline \
  --dataset data/all.jsonl \
  --train-output data/train.jsonl \
  --val-output data/val.jsonl \
  --checkpoint checkpoints/neural.pt \
  --log-dir logs/pipeline \
  --story-datasets data/test.json \
  --val-ratio 0.2 \
  --epochs 5 \
  --lr 3e-4
```

The pipeline writes the split datasets, stores training/evaluation metrics in `logs/pipeline`, and prints
Spearman correlations plus the model-found rate across the supplied evaluation stories. Afterwards you can demo the trained
policy inside the story experiments harness as shown above.

## Annotated literal datasets

The hand-authored stories under `data/` can be flattened into JSONL records that pair each sentence with
its resolved logical literal (including anaphora substitutions) via:

```bash
uv run python -m nls_python.model_generation.annotated_dataset \
  --inputs data/train.json data/val.json \
  --output data/annotated_literals.jsonl
```

Every line stores the story metadata, the raw sentence text, a `resolved_text` with pronouns replaced, the
logical literal string (with `NOT` prefixes for negations), and the literal/context history up to that
sentence. This file can supervise models that reason over the annotated data directly or report literal-level
metrics. Pass `--strict` if you prefer the command to fail whenever a sentence lacks an annotation (by default
they are skipped and reported). Top-k accuracy for heuristic experiments is available through
`neural_evaluation --top-k K`, the batch `policy_experiments` runner, and the automated `training_pipeline`.

### Seq2seq literal generation

Once you have `literals_{train,val,test}.jsonl`, you can fine-tune a pretrained seq2seq model (BART/T5/etc.)
to emit the target literal strings while respecting the variable role structure:

```bash
uv run python -m nls_python.model_generation.literal_seq2seq \
  --train data/literals_train.jsonl \
  --val data/literals_val.jsonl \
  --test data/literals_test.jsonl \
  --model-name facebook/bart-base \
  --output-dir experiments/literal_seq2seq \
  --epochs 5 \
  --batch-size 8 \
  --beam-size 5 \
  --top-k 5
```

The script logs training progress via HuggingFace's `Seq2SeqTrainer`, saves the fine-tuned checkpoint to
`experiments/literal_seq2seq/checkpoint`, and writes exact-match plus top-k accuracy for both validation and
test splits to `literal_metrics.json`. Generated inputs follow the `[DOMAIN] … [SYNOPSIS] … [CONTEXT_LITERAL] …`
template so they remain compatible with future symbolic/LLM components.

## Thesis experiment checklist

Follow this sequence to reproduce the complete evaluation pipeline described in the thesis.

1. **Ensure story splits exist (train/val/test).**  
   ```bash
   uv run python splitter.py           # writes data/train.json, data/val.json, data/test.json
   ```

2. **Generate literal JSONL files for each split.**  
   ```bash
   ./scripts/build_literal_datasets.sh      # or set STRICT=1 ./scripts/build_literal_datasets.sh
   ```
   (Add `--strict` if you want to fail on missing annotations instead of skipping them.)

3. **Distill every symbolic policy into a neural mimic and log top-k scores.**  
   ```bash
   ./scripts/run_policy_experiments.sh
   ```
   This script regenerates the datasets/checkpoints (using `--overwrite`), writes `experiments/policies/summary.json`,
   and produces `experiments/policies/metrics.png` with bar charts for model-found, top-k, and Spearman scores.

4. **Train the literal seq2seq model to emit structured predicates/roles.**  
   ```bash
   ./scripts/run_literal_seq2seq.sh              # set DEVICE=cpu to disable GPUs, or CUDA_VISIBLE_DEVICES=0 DEVICE=rocm
   ```
   The script fine-tunes BART (or a custom `MODEL_NAME`), saves checkpoints, enables TensorBoard logs under
   `experiments/literal_seq2seq/tb_logs`, and writes both `literal_metrics.json` and `literal_metrics.png`.

5. **Evaluate heuristics on the annotated test set.**  
   ```bash
   uv run python -m nls_python.model_generation.annotated_story_experiments \
     --input data/test.json \
     --base-policy CustomJava \
     --policies DFS BFS \
     --neural-checkpoint experiments/policies/CustomJava.pt \
     --output experiments/policies/annotated_test_results.txt
   ```
   This generates `HeuristicStatistics[...]` logs for every annotated story (base policy, comparison heuristics, and the
   neural mimic), matching the Java logging style without printing the full tableau dump.

With these artifacts generated, you can run `story_experiments` with any of the trained policies, compare the metrics
against the original heuristics, and insert the literal-model results alongside the symbolic top-k scores in the write-up.
