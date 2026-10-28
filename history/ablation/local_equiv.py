"""Local check that submission_ablate.py at its defaults trains exactly as the certified submission_adder.py, and that
every ablation switch runs (small: batch 32, well under 1 GB and a minute). Needs the H1 jsonl files (H1_DATA).
  python local_equiv.py
"""
import importlib.util, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
LR_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "learned_remainder")
sys.path[:0] = [os.path.join(LR_DIR, "chain_local", "stub"), os.path.join(LR_DIR, "adder_local")]
import torch
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch
from local_adder import load, batch


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, steps, B, seed, rows, sets=None):
    for k, v in (sets or {}).items():
        setattr(mod, k, v)
    torch.manual_seed(seed)
    m = mod.build_model(ModelSpec(vocab_size=17, max_seq_len=21, maximum_model_state_elements=500_000_000)).train()
    bundle = mod.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    g = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab = batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = mod.token_training_loss(TokenLossBatch(logits, lab, lab != -100, None, aux))
        bundle.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        if bundle.scheduler is not None:                    # as the evaluator: only when the submission has one
            bundle.scheduler.step()
        trace.append((float(loss.detach()), m.a.detach().clone(), m.b.detach().clone(), getattr(m.ctl, "phase", "-")))
    return trace


def main():
    rows = load("train", 4000)
    orig = os.path.join(LR_DIR, "submission_adder.py")
    abl = os.path.join(HERE, "submission_ablate.py")
    for label, sets, steps in (("defaults", {}, 90), ("phase B early", {"ID_STEPS": 20}, 60)):
        t0 = time.time()
        a = run(module(orig, "orig_" + label.replace(" ", "_")), steps, 32, 1, rows, sets)
        b = run(module(abl, "abl_" + label.replace(" ", "_")), steps, 32, 1, rows, sets)
        same = all(x[0] == y[0] and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) and x[3] == y[3]
                   for x, y in zip(a, b))
        print(f"{label}: {steps} steps, identical losses and parameters at every step: {same} "
              f"(phases {sorted(set(t[3] for t in a))}, {time.time() - t0:.1f}s)", flush=True)
    for sets in ({"ID_LINKS": "live"}, {"CE_SCOPE": "all"}, {"ADAM_RESET": False}, {"HARD_ANCHORS": False},
                 {"ID_AFTER_WARM": 0.0, "ADDER_WARM": 5}, {"TABLE_EVERY": 5, "LOG_EVERY": 10**9},
                 {"ID_AFTER_WARM": 0.0, "ID_GATE": True, "ADDER_WARM": 5}, {"ANCHOR_ORDER": 1.0},
                 {"ANCHOR_CHAIN": 1.0, "ANCHOR_WRAP": 0.0}, {"HARD_CIN": True}):
        tr = run(module(abl, "abl_x"), 12, 16, 1, rows, dict(sets, ID_STEPS=6))
        print("switch", sets, "runs: final loss", round(tr[-1][0], 4), "phase", tr[-1][3], flush=True)


if __name__ == "__main__":
    main()
