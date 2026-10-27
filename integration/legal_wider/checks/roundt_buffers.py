"""ROUNDT's rows in the buffers (WIDER.md §3). Random batches go through keep_counts, as combined_sp2's buffers part
sends them (counts 1, 2, 4, 8, 11, 3, 9, 0 and 22), beside a reference kept as the family file kept ROUNDT's rows
(Control.more: the newest ROUNDT rows of each count, in the order first seen). After every batch, the climb's rows as
climb_unit's own expression reads them from the buffers must equal the reference's counts T > 1, the first
ROUNDT_SLOTS of them. Then a fresh model loaded with the state dict: the same ROUNDT buffers and the same logits; and
the state's size (rule 5's count).

    cd cpu_first && COMBINED_FILE=integration/legal_wider/submission_stage3_wider.py PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/legal_wider/checks/roundt_buffers.py
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[2]
sys.path.insert(0, str(CPU / "integration" / "combined_sp2" / "checks"))
import local_combined as LC  # noqa: E402  (its FILE is COMBINED_FILE)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def main():
    M = LC.load([("MODE", '"designed"')])
    expr = re.search(r"ctl\.climb\.more = (\[.*?\])\n", open(LC.FILE).read(), re.S).group(1)   # climb_unit's own
    print(f"{LC.FILE}: ROUNDT {M.ROUNDT}, ROUNDT_SLOTS {M.ROUNDT_SLOTS}; climb_unit: {' '.join(expr.split())}", flush=True)
    m = LC.designed_model(M)
    m.ctl.phase, m.ctl.io = "C", 5
    gen, K, W = torch.Generator().manual_seed(5), M.K, M.W
    ref, fails, kept = {}, 0, None
    for step in range(300):
        nb = int(torch.randint(1, 200, (1,), generator=gen))
        T = torch.tensor([1, 2, 4, 8, 11, 3, 9, 0, 22])[torch.randint(0, 9 if step > 100 else 4, (nb,), generator=gen)]
        Cd = torch.stack([T % 10, T // 10], 1)
        Nd, Xd, yd = (torch.randint(0, 10, (nb, W), generator=gen) for _ in range(3))
        m.t_readings = lambda aux_, Cd=Cd: (torch.ones(Cd.shape[0], dtype=torch.bool), None, None, None, None, Cd, Cd)
        m.keep_counts({"Nd": Nd, "Xd": Xd, "Cd": Cd}, yd)
        slot = (Cd * 10 ** torch.arange(K - 1, -1, -1)).sum(1)          # the family file's lines, as they were
        new = torch.stack([t.long() for t in (Nd, Xd, F.pad(Cd, (0, W - K)), yd)], 1)
        for s in torch.unique(slot).tolist():
            ref[s] = torch.cat([ref.get(s, new[:0]), new[slot == s]])[-M.ROUNDT:]
        want = [(T_, (r[:, 0], r[:, 1], r[:, 3])) for r in ref.values() for T_ in [int(M.count_of(r[:1, 2, :K]))]
                if T_ > 1][:M.ROUNDT_SLOTS]
        got = eval(expr, {"self": m, "count_of": M.count_of, "K": K, "ROUNDT_SLOTS": M.ROUNDT_SLOTS})
        same = len(got) == len(want) and all(a[0] == b[0] and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
                                             for a, b in zip(got, want))
        fails += not same
        kept = [(T_, int(r[0].shape[0])) for T_, r in got]
    print(f"ROUNDT's rows from the buffers equal the family file's dict after each of 300 random batches: "
          f"{'yes' if not fails else f'NO ({fails} batches differ)'}; kept (T, rows): {kept}", flush=True)
    from benchmark import ModelSpec
    import local_check as lc
    cfg = json.load(open(os.path.join(LC.DATA, "dataset_config.json")))
    torch.manual_seed(123)
    m2 = M.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
    sd = m.state_dict()
    for k in ("round_ops", "round_src"):                              # (they grow when the round is found)
        setattr(m2, k, torch.zeros_like(sd[k]))
    m2.load_state_dict(sd)
    ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(LC.DATA, "depth_t_4", 6))
    with torch.no_grad():
        a, b = m(ids, am)[0], m2(ids, am)[0]
    bufs = all(torch.equal(getattr(m, k), getattr(m2, k)) for k in ("more_rows", "more_n", "more_s"))
    n_par, n_buf = sum(p.numel() for p in m.parameters()), sum(b_.numel() for b_ in m.buffers())
    fresh = M.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000))
    print(f"a fresh model loaded with the state dict: ROUNDT's buffers equal: {bufs}; the same logits: "
          f"{bool(torch.equal(a, b))}. State: {n_par:,} parameters, {n_buf:,} buffer numbers here; at the start "
          f"{sum(p.numel() for p in fresh.parameters()) + sum(b_.numel() for b_ in fresh.buffers()):,} numbers, of "
          f"which ROUNDT's {sum(getattr(fresh, k).numel() for k in ('more_rows', 'more_n', 'more_s')):,}", flush=True)


if __name__ == "__main__":
    main()
