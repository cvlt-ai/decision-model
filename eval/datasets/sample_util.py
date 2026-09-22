"""Deterministic sampling helpers shared by adapters."""

from __future__ import annotations

import numpy as np
from datasets import load_dataset

# Pinned per-family seeds; changing one invalidates the frozen holdout on purpose.
SEEDS = {
    "boolq": 20260922,
    "banking77": 20260923,
    "mmlu_pro": 20260924,
    "injection": 20260925,
    "go_emotions": 20260926,
    "pubhealth": 20260927,
    "negation": 20260928,
    "baserate": 20260929,
}


def get_split(hf_id: str, split: str, config: str | None = None):
    """Load one split. Never 'train' from here: eval code must not touch train data."""
    if split == "train":
        raise ValueError("eval adapters must not load train splits")
    if config:
        return load_dataset(hf_id, config, split=split)
    return load_dataset(hf_id, split=split)


def sample(ds, n: int | None, seed_key: str):
    """Deterministic shuffle-then-take. Returns (row_list, actual_n)."""
    rng = np.random.default_rng(SEEDS[seed_key])
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    if n is not None:
        idx = idx[: min(n, len(idx))]
    return [ds[int(i)] for i in idx]
