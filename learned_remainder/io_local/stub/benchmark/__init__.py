"""Local stand-in for the evaluator's public API (only what the submission imports)."""
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int
    max_seq_len: int
    maximum_model_state_elements: int


@dataclass(frozen=True)
class OptimizerSpec:
    training_time_seconds: float
    device_type: str


@dataclass
class OptimizerBundle:
    optimizer: Any
    scheduler: Any = None


@dataclass(frozen=True)
class TokenLossBatch:
    logits: Any
    labels: Any
    valid_mask: Any
    target_positions: Any
    auxiliary: Any


@dataclass(frozen=True)
class Submission:
    build_model: Callable
    build_optimizer: Callable
    training_loss: Optional[Callable] = None
    token_training_loss: Optional[Callable] = None
    batch_size: Optional[int] = None
    max_steps: Optional[int] = None
    eval_batch_size: Optional[int] = None


def assert_model_state(model, spec):
    n = sum(p.numel() for p in model.parameters())
    assert n <= spec.maximum_model_state_elements
    return n
