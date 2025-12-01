import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")  # Ensure non-interactive backend for CLI/tests.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
from nls_python.typing_utils import typechecked

from .narratives import NarrativeStep
from .neural import NeuralPolicyConfig, NarrativeNeuralNetwork

import torch
from torch import Tensor
from torch.optim import AdamW


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for neural training. Install NLS-Python with the "
            "cpu/cu126/rocm extra."
        )


logger = logging.getLogger(__name__)


@typechecked
def _dict_to_step(raw: Dict[str, Any]) -> NarrativeStep:
    roles = raw.get("roles") or {}
    anaphora = raw.get("anaphora") or {}
    semantic_types = raw.get("semantic_types") or {}
    if not isinstance(roles, Mapping):
        roles = {}
    if not isinstance(anaphora, Mapping):
        anaphora = {}
    if not isinstance(semantic_types, Mapping):
        semantic_types = {}
    return NarrativeStep(
        index=int(raw["index"]),
        verb=str(raw["verb"]),
        negated=bool(raw.get("negated", False)),
        text=str(raw.get("text", "")),
        roles=dict(roles),
        anaphora=dict(anaphora),
        semantic_types=dict(semantic_types),
    )


def _rank_to_target(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - rank / max(total - 1, 1)


def _load_records(dataset_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Dataset {dataset_path} is empty.")
    return records


def _training_step(
    network: NarrativeNeuralNetwork,
    optimizer: AdamW,
    record: Dict[str, Any],
    *,
    loss_fn,
) -> Tuple[float, float]:
    _require_torch()
    network.train()
    optimizer.zero_grad()
    steps = [_dict_to_step(step) for step in record["narrative_steps"]]
    raw_score: Tensor = network(steps)
    score = torch.sigmoid(raw_score)
    target_value = _rank_to_target(int(record["rank"]), int(record["total_models"]))
    target = torch.tensor(target_value, dtype=torch.float32, device=score.device)
    loss = loss_fn(score, target)
    loss.backward()
    optimizer.step()
    predicted_positive = bool(score.item() >= 0.5)
    target_positive = bool(target_value >= 0.5)
    correct = 1.0 if predicted_positive == target_positive else 0.0
    return float(loss.item()), correct


def train_from_dataset(
    dataset_path: Path,
    *,
    output_path: Path,
    epochs: int = 3,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    config: Optional[NeuralPolicyConfig] = None,
    log_dir: Optional[Path] = None,
) -> Path:
    _require_torch()
    torch.manual_seed(seed)
    random.seed(seed)
    records = _load_records(dataset_path)
    policy_config = config or NeuralPolicyConfig(device=device)
    network = NarrativeNeuralNetwork(policy_config)
    optimizer = AdamW(network.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    epoch_losses: List[float] = []
    epoch_accuracies: List[float] = []
    for epoch in range(epochs):
        random.shuffle(records)
        total_loss = 0.0
        total_correct = 0.0
        for record in records:
            loss_value, correct = _training_step(
                network, optimizer, record, loss_fn=loss_fn
            )
            total_loss += loss_value
            total_correct += correct
        avg_loss = total_loss / len(records)
        accuracy = total_correct / len(records)
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        logger.info(
            "Epoch %d/%d - loss=%.4f accuracy=%.3f",
            epoch + 1,
            epochs,
            avg_loss,
            accuracy,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(policy_config),
            "state_dict": network.state_dict(),
        },
        output_path,
    )
    if log_dir and plt is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        epochs_range = range(1, epochs + 1)
        plt.figure()
        plt.plot(epochs_range, epoch_losses, marker="o", label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Neural Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / "loss.png")
        plt.close()

        plt.figure()
        plt.plot(
            epochs_range,
            epoch_accuracies,
            marker="o",
            color="green",
            label="Accuracy",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.title("Neural Training Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / "accuracy.png")
        plt.close()

        metrics_path = log_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "loss": epoch_losses,
                    "accuracy": epoch_accuracies,
                },
                handle,
                indent=2,
            )
    elif log_dir:
        logger.warning(
            "matplotlib is not available; skipping training plots for %s", log_dir
        )
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the narrative neural heuristic."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination checkpoint (.pt).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional directory for training plots/logs.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    train_from_dataset(
        args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
    )
    logger.info("Saved checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
