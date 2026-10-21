"""Local check that submission_combined_ablate.py at its defaults trains exactly as the certified
cpu_first/learned_remainder/submission_combined.py: identical losses and parameters at every step (batch 16, 30 steps;
small: well under 1 GB and a minute), and that its two switches run.
  python local_equiv_combined.py
"""
import importlib.util, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
LR_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "learned_remainder")
sys.path[:0] = [os.path.join(LR_DIR, "chain_local", "stub"), os.path.join(LR_DIR, "combined_local"), LR_DIR]
import torch
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch
from local_combined import load, batch


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, steps, B, seed, rows, sets=None):
    for k, v in (sets or {}).items():
        setattr(mod, k, v)
    mod.LOG_EVERY = 10 ** 9
    torch.manual_seed(seed)
    m = mod.build_model(ModelSpec(vocab_size=17, max_seq_len=21, maximum_model_state_elements=500_000_000)).train()
    bundle = mod.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    g = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab, tpos = batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = mod.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        bundle.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()
        trace.append((float(loss.detach()), [p.detach().clone() for p in m.parameters()]))
    return trace


def main():
    rows = load("train", 4000)
    t0 = time.time()
    a = run(module(os.path.join(LR_DIR, "submission_combined.py"), "orig"), 30, 16, 1, rows)
    b = run(module(os.path.join(HERE, "submission_combined_ablate.py"), "abl"), 30, 16, 1, rows)
    same = all(x[0] == y[0] and all(torch.equal(p, q) for p, q in zip(x[1], y[1])) for x, y in zip(a, b))
    print(f"defaults: 30 steps, identical losses and parameters at every step: {same} ({time.time() - t0:.1f}s)")
    for sets in ({"ID_AFTER_WARM": 0.0, "ADDER_WARM": 3, "WARM_TO_FACTS": False}, {"TABLE_EVERY": 4}):
        tr = run(module(os.path.join(HERE, "submission_combined_ablate.py"), "abl_x"), 8, 16, 1, rows, sets)
        print("switch", sets, "runs: final loss", round(tr[-1][0], 4), flush=True)


if __name__ == "__main__":
    main()
