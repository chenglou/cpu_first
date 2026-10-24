"""H1's rows for the offline checks (integration/self_practice/core.py's two helpers, which import the old libcore):
the released training or depth split, as digit tensors read from the decimal strings, never computed."""
import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
H1_DIR = os.environ.get("H1_DATA", str(HERE.parents[2] / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


def load_rows(n, seed, T=None, split="train"):
    rows = [json.loads(l) for l in open(os.path.join(H1_DIR, split + ".jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits(rows, key, L_=8):
    """A field's decimal strings as digit tensors [n, L] (LSB first), read, never computed."""
    return torch.tensor([[int(ch) for ch in reversed(str(r[key]))][:L_] + [0] * (L_ - min(L_, len(str(r[key]))))
                         for r in rows])
