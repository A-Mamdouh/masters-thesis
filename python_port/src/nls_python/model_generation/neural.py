import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, TYPE_CHECKING

from nls_python.typing_utils import typechecked

from .heuristics import ModelFeatures
from .narratives import NarrativeStep
from .policies import SearchPolicy
from .tableau_model import TableauModel

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = SimpleNamespace(Module=object)  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for neural policies. Install NLS-Python with the "
            "extra dependencies (cpu/cu126/rocm) or provide torch in the environment."
        )


@typechecked
@dataclass
class NeuralPolicyConfig:
    vocab_size: int = 4096
    embedding_dim: int = 256
    attention_heads: int = 4
    context_size: int = 8
    device: str = "cpu"
    dropout: float = 0.1


class HashedTextEncoder(nn.Module):
    """Lightweight text encoder that maps tokens to embeddings via hashing."""

    def __init__(self, config: NeuralPolicyConfig) -> None:
        _require_torch()
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.device = torch.device(config.device)
        self.to(self.device)

    def forward(self, tokens: Sequence[str]) -> Tensor:  # type: ignore[override]
        if not tokens:
            tokens = ("<EMPTY>",)
        indices = torch.tensor(
            [self._hash(token) for token in tokens],
            dtype=torch.long,
            device=self.device,
        )
        embeddings = self.embedding(indices)
        return embeddings.mean(dim=0)

    def _hash(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest, "big") % self.vocab_size


class NarrativeNeuralNetwork(nn.Module):
    """Sequence model that consumes narrative steps and outputs a scalar score."""

    def __init__(self, config: NeuralPolicyConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.text_encoder = HashedTextEncoder(config)
        self.context_tokens = nn.Parameter(
            torch.randn(
                (config.context_size, config.embedding_dim),
                device=self.text_encoder.device,
            )
        )
        self.sentence_projector = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.attention_heads,
            batch_first=True,
            dropout=config.dropout,
        )
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.score_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, 1),
        )
        self.dropout = nn.Dropout(config.dropout)

    @typechecked
    def forward(self, steps: Sequence[NarrativeStep]) -> Tensor:  # type: ignore[override]
        context = self.context_tokens.unsqueeze(0)
        for step in steps:
            step_embedding = self._encode_step(step).unsqueeze(0).unsqueeze(0)
            attn_input = torch.cat([step_embedding, context], dim=1)
            attn_out, _ = self.attention(
                attn_input, attn_input, attn_input, need_weights=False
            )
            updated = attn_out[:, 1:, :]
            context = self.norm(updated)
        summary = context[:, -1, :]
        summary = self.dropout(summary)
        return self.score_head(summary).squeeze(0).squeeze(-1)

    @typechecked
    def _encode_step(self, step: NarrativeStep) -> Tensor:
        tokens = [f"verb::{step.verb}", f"idx::{step.index}"]
        for role, value in sorted(step.roles.items()):
            resolved = step.anaphora.get(value, value)
            tokens.append(f"role::{role}={resolved}")
            if resolved != value:
                tokens.append(f"role_raw::{role}={value}")
            sem_key = step.semantic_types.get(resolved) or step.semantic_types.get(value)
            if sem_key:
                tokens.append(f"sem::{role}={sem_key}")
        if step.negated:
            tokens.append("modifier::negated")
        if step.text:
            tokens.append(f"text::{step.text.lower()}")
        embedding = self.text_encoder(tokens)
        return self.sentence_projector(embedding)


class NarrativeNeuralScorer:
    """Callable wrapper suitable for SearchPolicy cost functions."""

    def __init__(
        self, network: NarrativeNeuralNetwork, *, higher_is_better: bool = False
    ) -> None:
        self._network = network
        self._sign = -1.0 if higher_is_better else 1.0
        self._network.eval()

    def __call__(self, model: TableauModel, features: ModelFeatures) -> float:
        _require_torch()
        narrative = model.get_narrative()
        steps = narrative.steps
        if not steps:
            return float(features.model_id)
        with torch.inference_mode():
            score = self._network(steps)
        return self._sign * float(score.item())


@typechecked
def make_story_neural_policy(
    name: str,
    network: NarrativeNeuralNetwork,
    *,
    drain_size: int = -1,
    higher_is_better: bool = False,
) -> SearchPolicy:
    scorer = NarrativeNeuralScorer(network, higher_is_better=higher_is_better)
    return SearchPolicy(name=name, cost_function=scorer, drain_size=drain_size)


@typechecked
def load_story_neural_policy(
    name: str,
    checkpoint_path: Path | str,
    *,
    drain_size: int = -1,
    device: str | None = None,
    higher_is_better: bool = False,
) -> SearchPolicy:
    _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config_dict: Dict[str, Any] = checkpoint.get("config", {})
    if device:
        config_dict["device"] = device
    config = NeuralPolicyConfig(**config_dict)
    network = NarrativeNeuralNetwork(config)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint is missing 'state_dict'.")
    network.load_state_dict(state_dict)
    return make_story_neural_policy(
        name,
        network,
        drain_size=drain_size,
        higher_is_better=higher_is_better,
    )


__all__ = [
    "HashedTextEncoder",
    "NarrativeNeuralNetwork",
    "NarrativeNeuralScorer",
    "NeuralPolicyConfig",
    "make_story_neural_policy",
    "load_story_neural_policy",
]
