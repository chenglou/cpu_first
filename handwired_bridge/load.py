"""Load handwired's compiled tick program into PyTorch and check it, question by question and cell by cell.

`js/compile.ts` writes, per size, the program that handwired's `compileTensorProgram` made from the tick of
`js/machine.ts`: layers of sparse affine rows, some of them gated by a ReLU (ReGLU), each layer reading any earlier
value. This module runs those layers as they are (no knowledge of what they compute), with the host loop around them:
`depth` evaluations, each writing the state cells' next values back into the source table.

The checks (`python -m handwired_bridge.load`, record `records/compile.json`), at every size of the attention bridge's
compile table:
- the JS layout is the attention bridge's `grid.Layout` (kinds, input wiring, ports, answer cells, ticks);
- every question (a random 20,000 above that) comes out exactly v mod N, in float64 and in float32;
- every cell of the grid holds the machine's own value (`grid.machine_reference`), digit and borrow;
- the same from random one-hot scratch states.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from bridge import grid as G

HERE = Path(__file__).resolve().parent
COMPILED = HERE / "records" / "compiled"

SIZES = [(3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 2, 4), (3, 3, 2), (3, 3, 3), (2, 3, 3), (2, 4, 4), (4, 2, 2), (4, 2, 3),
         (5, 2, 2), (4, 3, 2), (5, 3, 2), (3, 4, 3)]


def record(B, w, k):
    return json.loads((COMPILED / f"B{B}w{w}k{k}.json").read_text())


class Program:
    """A compiled tick program as PyTorch tensors: per layer the rows of its sparse matrix as a dense matrix over the
    values the layer reads, its bias, and which output rows are gated (output * relu(gate row))."""

    def __init__(self, rec, dtype=torch.float64):
        p = rec["program"]
        self.rec, self.dtype = rec, dtype
        self.source_size, self.value_count = p["sourceSize"], p["valueCount"]
        self.layers = []
        for layer in p["layers"]:
            rows = len(layer["bias"])
            W = torch.zeros(rows, len(layer["inputIndices"]), dtype=dtype)
            offsets = layer["rowOffsets"]
            for r in range(rows):
                for e in range(offsets[r], offsets[r + 1]):
                    W[r, layer["columns"][e]] += layer["weights"][e]
            self.layers.append({
                "inputs": torch.tensor(layer["inputIndices"]), "W": W,
                "bias": torch.tensor(layer["bias"], dtype=dtype), "out": layer["outputSize"],
                "gates": torch.tensor(layer["gateRows"], dtype=torch.long),
            })
        order = p["wireOrder"]
        where = {wire: i for i, wire in enumerate(order)}
        self.root_pos = torch.tensor([where[r] for r in rec["roots"]])
        self.state = torch.tensor(rec["stateWires"])
        self.lay = rec["layout"]
        self.digit = torch.tensor([cell["digit"] for cell in rec["grid"]])          # [L, B] source wires
        self.borrow = torch.tensor([cell["borrow"] for cell in rec["grid"]])        # [L, 2]

    def evaluate(self, source):
        """One evaluation: source [X, sourceSize] -> the roots' values [X, roots] (the state cells' next values)."""
        X = source.shape[0]
        packed = torch.zeros(X, self.value_count, dtype=self.dtype)
        packed[:, :self.source_size] = source
        pos = self.source_size
        for layer in self.layers:
            proj = packed[:, layer["inputs"]] @ layer["W"].T + layer["bias"]
            out = proj[:, :layer["out"]].clone()
            g = layer["gates"]
            if len(g):
                out[:, g] = out[:, g] * torch.relu(proj[:, layer["out"]:layer["out"] + len(g)])
            packed[:, pos:pos + layer["out"]] = out
            pos += layer["out"]
        return packed[:, self.root_pos]

    def source(self, states):
        """A source table [X, sourceSize] from grid states [X, L, B + 2] (digit one-hot, borrow one-hot)."""
        X, L, _ = states.shape
        B = self.digit.shape[1]
        src = torch.zeros(X, self.source_size, dtype=self.dtype)
        src[:, self.digit.flatten()] = states[..., :B].reshape(X, -1).to(self.dtype)
        src[:, self.borrow.flatten()] = states[..., B:].reshape(X, -1).to(self.dtype)
        return src

    def states(self, source):
        B = self.digit.shape[1]
        return torch.cat([source[:, self.digit], source[:, self.borrow]], -1)          # [X, L, B + 2]

    def run(self, states, ticks=None):
        """The host loop: `ticks` (default the layout's depth) evaluations, each writing the state cells back."""
        src = self.source(states)
        for _ in range(ticks or self.lay["depth"]):
            src[:, self.state] = self.evaluate(src)
        return self.states(src)


def same_layout(rec, lay: G.Layout):
    js = rec["layout"]
    ports = np.array(js["ports"])
    return bool(js["L"] == lay.L and js["depth"] == lay.depth and (np.array(js["kind"]) == lay.kind).all()
                and (np.array(js["src"]) == lay.src).all() and (ports == lay.comp).all()
                and (np.array(js["answer"]) == lay.answer).all() and (np.array(js["row"]) == lay.row).all()
                and (np.array(js["col"]) == lay.col).all())


def check(B, w, k, sample=20000, chunk=2000, seed=0):
    rec = record(B, w, k)
    lay = G.Layout(B, w, k)
    data = G.Data(B, w, k, sample=sample, seed=1)
    out = {"size": [B, w, k], "questions": int(data.X), "same_layout": same_layout(rec, lay), "counts": rec["counts"],
           "js_checks": rec["checks"]}
    dig_ref, bor_ref = G.machine_reference(lay, data)
    gen = torch.Generator().manual_seed(seed)
    for dtype in (torch.float64, torch.float32):
        prog = Program(rec, dtype)
        right = cells_right = rand_right = 0
        for i in range(0, data.X, chunk):
            idx = np.arange(i, min(i + chunk, data.X))
            d = data.subset(idx)
            S0 = G.data_inputs(lay, d, dtype)
            S = prog.run(S0)
            ans = S[:, lay.answer, :B].argmax(-1)
            right += int((ans == d.yd).all(-1).sum())
            dig = S[..., :B].argmax(-1).numpy()
            bor = S[..., B:].argmax(-1).numpy()
            exact_bits = bool(((S == 0) | (S == 1)).all())
            cells_right += int(((dig == dig_ref[idx]) & (bor == bor_ref[idx])).all(1).sum()) if exact_bits else 0
            # random one-hot scratch states
            Sr = S0.clone()
            scratch = torch.from_numpy(~lay.is_input)
            n = int(scratch.sum())
            rd = torch.randint(B, (len(idx), n), generator=gen)
            rb = torch.randint(2, (len(idx), n), generator=gen)
            Sr[:, scratch] = torch.cat([torch.nn.functional.one_hot(rd, B), torch.nn.functional.one_hot(rb, 2)],
                                       -1).to(dtype)
            Sr = prog.run(Sr)
            rand_right += int((Sr[:, lay.answer, :B].argmax(-1) == d.yd).all(-1).sum())
        tag = "f64" if dtype == torch.float64 else "f32"
        out[f"answers_{tag}"] = right
        out[f"cells_{tag}"] = cells_right
        out[f"random_start_{tag}"] = rand_right
    out["exact"] = bool(out["same_layout"] and all(out[f"{m}_{t}"] == data.X for m in ("answers", "cells", "random_start")
                                                   for t in ("f64", "f32")))
    return out


def main():
    rows = []
    for B, w, k in SIZES:
        t0 = time.time()
        r = check(B, w, k)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"({B},{w},{k}) questions {r['questions']} layout {'=' if r['same_layout'] else '!='} bridge | "
              f"answers {r['answers_f64']}/{r['answers_f32']} cells {r['cells_f64']}/{r['cells_f32']} "
              f"random start {r['random_start_f64']}/{r['random_start_f32']} | ops {r['counts']['liveOperations']} "
              f"weights {r['counts']['storedWeights']} | {'EXACT' if r['exact'] else 'WRONG'} {r['seconds']}s",
              flush=True)
    (HERE / "records" / "compile.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
