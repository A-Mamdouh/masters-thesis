"""Seq2seq training pipeline for generating logical literals from annotated stories."""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional torch dependency
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

if TYPE_CHECKING:
    from torch import device as TorchDevice
else:
    TorchDevice = Any

from nls_python.typing_utils import typechecked

try:  # pragma: no cover - optional transformers dependency
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except ImportError:  # pragma: no cover
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    DataCollatorForSeq2Seq = None
    Seq2SeqTrainer = None
    Seq2SeqTrainingArguments = None


logger = logging.getLogger(__name__)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for literal seq2seq training. Install torch or "
            "use the pre-generated literal datasets without training."
        )


def _require_transformers() -> None:
    if AutoTokenizer is None:
        raise RuntimeError(
            "HuggingFace Transformers is required for literal seq2seq training. "
            "Install it (e.g., `uv pip install transformers`)."
        )


def _prepare_device(device_request: str) -> Tuple[TorchDevice, bool]:
    """Resolve device argument, supporting CPU, CUDA, and ROCm (HIP)."""
    _require_torch()
    normalized = (device_request or "auto").lower()
    has_cuda = torch.cuda.is_available()
    if normalized in {"auto", "cuda", "gpu"}:
        if has_cuda:
            return torch.device("cuda"), True
        if normalized != "auto":
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cpu"), False
    if normalized in {"rocm", "hip"}:
        if has_cuda:
            # ROCm builds expose a CUDA-like device interface.
            os.environ.setdefault("PYTORCH_ROCM_ENABLED", "1")
            return torch.device("cuda"), True
        raise RuntimeError("ROCm requested but no compatible GPU is available.")
    if normalized == "cpu":
        return torch.device("cpu"), False
    # Fallback: trust torch to interpret the string.
    return torch.device(normalized), normalized != "cpu"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_roles(roles: Dict[str, str]) -> str:
    if not roles:
        return ""
    return " ".join(f"{role}:{value}" for role, value in roles.items())


def build_input_text(record: Dict[str, Any]) -> str:
    synopsis = record.get("story_synopsis") or ""
    domain = record.get("story_domain") or "unknown"
    parts = [
        f"[DOMAIN] {domain}",
        f"[SYNOPSIS] {synopsis}",
    ]
    for literal in record.get("context_literals", []) or []:
        parts.append(f"[CONTEXT_LITERAL] {literal}")
    resolved = record.get("resolved_text") or record.get("text") or ""
    parts.append(f"[SENTENCE] {resolved}")
    roles = record.get("roles") or {}
    role_block = _format_roles(roles)
    if role_block:
        parts.append(f"[ROLES] {role_block}")
    return " ".join(parts).strip()


def build_target_text(record: Dict[str, Any]) -> str:
    literal = record.get("literal")
    if literal is None:
        return ""
    if isinstance(literal, str):
        return literal
    if isinstance(literal, (list, tuple)):
        return " || ".join(str(item) for item in literal)
    return str(literal)


def load_literal_records(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path:
            continue
        records.extend(_read_jsonl(path))
    if not records:
        raise ValueError("No records found in the provided paths.")
    return records


@dataclass(frozen=True)
class LiteralExample:
    input_text: str
    target_text: str


def build_examples(records: Iterable[Dict[str, Any]]) -> List[LiteralExample]:
    return [
        LiteralExample(
            input_text=build_input_text(record),
            target_text=build_target_text(record),
        )
        for record in records
    ]


class LiteralSeq2SeqDataset(Dataset):
    """Lightweight torch dataset that tokenizes on-the-fly."""

    def __init__(
        self,
        examples: Sequence[LiteralExample],
        tokenizer,
        *,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        _require_torch()
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        inputs = self.tokenizer(
            example.input_text,
            max_length=self.max_input_length,
            truncation=True,
        )
        targets = self.tokenizer(
            example.target_text,
            max_length=self.max_target_length,
            truncation=True,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs


def _batched(iterable: Sequence[LiteralExample], batch_size: int) -> Iterable[List[LiteralExample]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def evaluate_literal_model(
    model,
    tokenizer,
    examples: Sequence[LiteralExample],
    *,
    max_input_length: int,
    max_target_length: int,
    beam_size: int,
    top_k: int,
    batch_size: int,
    device: TorchDevice,
) -> Dict[str, float]:
    _require_torch()
    model.eval()
    total = len(examples)
    if total == 0:
        return {"exact_match": 0.0, "topk_accuracy": 0.0}
    exact = 0
    top_hits = 0
    num_beams = max(beam_size, top_k)
    num_return = max(1, min(top_k, num_beams))
    with torch.no_grad():
        for batch_examples in _batched(list(examples), batch_size):
            texts = [ex.input_text for ex in batch_examples]
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_return,
                max_length=max_target_length,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for idx, example in enumerate(batch_examples):
                span = decoded[idx * num_return : (idx + 1) * num_return]
                gold = example.target_text.strip()
                if span:
                    if span[0].strip() == gold:
                        exact += 1
                    if any(candidate.strip() == gold for candidate in span):
                        top_hits += 1
    return {
        "exact_match": exact / total,
        "topk_accuracy": top_hits / total,
    }


@typechecked
def train_literal_seq2seq(
    *,
    train_paths: Sequence[Path],
    val_paths: Sequence[Path] | None,
    test_paths: Sequence[Path] | None,
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_input_length: int,
    max_target_length: int,
    beam_size: int,
    top_k: int,
    device: str,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    _require_torch()
    _require_transformers()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device, use_cuda = _prepare_device(device)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(torch_device)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = build_examples(load_literal_records(train_paths))
    val_examples = (
        build_examples(load_literal_records(val_paths))
        if val_paths
        else []
    )

    train_dataset = LiteralSeq2SeqDataset(
        train_examples,
        tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )
    val_dataset = (
        LiteralSeq2SeqDataset(
            val_examples,
            tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )
        if val_examples
        else None
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    logging_dir = output_dir / "tb_logs"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch" if val_dataset is not None else "no",
        predict_with_generate=val_dataset is not None,
        logging_steps=50,
        save_strategy="epoch",
        seed=seed,
        fp16=use_cuda,
        no_cuda=not use_cuda,
        report_to=["tensorboard"],
        run_name="literal_seq2seq",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    eval_metrics: Dict[str, Dict[str, float]] = {}
    if val_examples:
        eval_metrics["val"] = evaluate_literal_model(
            model,
            tokenizer,
            val_examples,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            beam_size=beam_size,
            top_k=top_k,
            batch_size=eval_batch_size,
            device=torch_device,
        )
        logger.info("Validation metrics: %s", eval_metrics["val"])
    if test_paths:
        test_examples = build_examples(load_literal_records(test_paths))
        eval_metrics["test"] = evaluate_literal_model(
            model,
            tokenizer,
            test_examples,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            beam_size=beam_size,
            top_k=top_k,
            batch_size=eval_batch_size,
            device=torch_device,
        )
        logger.info("Test metrics: %s", eval_metrics["test"])
    trainer.save_model(str(output_dir / "checkpoint"))
    tokenizer.save_pretrained(output_dir / "checkpoint")
    metrics_path = output_dir / "literal_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, indent=2)
    logger.info("Saved metrics to %s", metrics_path)
    return eval_metrics


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a seq2seq model to emit logical literals."
    )
    parser.add_argument(
        "--train",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to literal JSONL files for training.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        nargs="+",
        default=None,
        help="Optional path(s) for validation data.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        nargs="+",
        default=None,
        help="Optional path(s) for test evaluation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/bart-base",
        help="Pretrained seq2seq checkpoint (default: facebook/bart-base).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store checkpoints and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device selection: auto, cpu, cuda, or rocm.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    train_literal_seq2seq(
        train_paths=args.train,
        val_paths=args.val,
        test_paths=args.test,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        beam_size=args.beam_size,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
