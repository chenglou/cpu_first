"""Summaries of submission_io.py run records (cpu_first/runs/learned_remainder/<name>.json): the evaluator's result,
the subtractor audit (summarize.py), the I/O schedule, and the snapped I/O the submission printed at its first
evaluation batch (IO lines), checked here, outside the submission, against the true reading of every depth prompt.
    python summarize_io.py NAME [NAME ...]
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import torch
import summarize as sm
from local_io import load, batch


def snapped(stdout):
    lines = stdout.splitlines()
    io = {}
    for l in lines:
        m = re.match(r"IO field=(\S+) anchor=(\S+) step=(\S+)", l)
        if m:
            io[m.group(1)] = (m.group(2), int(m.group(3)))
        if l.startswith("IO code="):
            io["code"] = {int(a): int(b) for a, b in (kv.split(":") for kv in l[8:].split(","))}
        if l.startswith("IO writer="):
            io["wcode"] = {int(a): int(b) for a, b in (kv.split(":") for kv in l[10:].split(","))}
        if l.startswith("IO digit_tokens="):
            io["digits"] = json.loads(l.split("=", 1)[1].split(" reading")[0])
            io["reading"] = l.split("reading=")[1].split(" rows_with")[0]
    sched = [l for l in lines if l.startswith("LR io_stage") or l.startswith("LR switch") or l.startswith("LR frozen")]
    return io, sched


def check_reader(io, split):
    """Read every prompt of a split with the printed snapped I/O; compare with the jsonl's N, x, T, and write the
    true answer with the printed writer, compared at the target positions."""
    ids, lab, num = load(split)
    ids, mask, lab, tp = batch(ids, lab)
    end = mask.sum(1)
    X, L = ids.shape
    e = torch.cat([ids, torch.full((X, 1), -2)], 1)
    e[torch.arange(L + 1)[None].expand(X, -1) == end[:, None]] = 17          # END at the prompt's end
    e = torch.nn.functional.pad(e, (9, 9), value=-1)
    isd = torch.zeros(19, dtype=torch.bool); isd[io["digits"]] = True
    val = torch.zeros(19, dtype=torch.long)
    for t, v in io["code"].items():
        val[t] = v

    def read(field, n):
        a, s = io[field]
        a = 17 if a == "END" else int(a)
        p = torch.arange(e.shape[1])[None]
        okp = (p >= 9) & (p <= 9 + end[:, None])
        q = torch.where(okp & (e == a), p, 10 ** 6).min(1).values
        pos = q[:, None] + (torch.arange(n)[None] + 1) * s
        t = e.gather(1, pos.clamp(0, e.shape[1] - 1))
        live = isd[t.clamp(min=0)] & (t >= 0)
        live = live.long().cumprod(-1).bool()
        return torch.where(live, val[t.clamp(min=0)], 0), pos

    P10 = 10 ** torch.arange(8)
    out = {}
    for field, n, col in (("N", 8, 0), ("x", 8, 1), ("T", 2, 2)):
        d, _ = read(field, n)
        out[field] = int(((d * P10[:n]).sum(1) == num[:, col]).sum())
    _, pos = read("writer", 8)
    wtok = torch.tensor([io["wcode"][v] for v in range(10)])
    yd = (num[:, 3][:, None] // P10) % 10
    o = torch.full_like(e, -1).scatter_(1, pos.clamp(0, e.shape[1] - 1), wtok[yd])
    valid = lab != -100
    got = o.gather(1, torch.where(valid, tp + 9, 0))
    out["writer"] = int((torch.where(valid, got, lab) == lab).all(1).sum())
    out["rows"] = X
    out["8-digit N"] = int((num[:, 0] >= 10 ** 7).sum())
    return out


def main(names):
    for name in names:
        rec = json.loads((sm.RUNS / f"{name}.json").read_text())
        res, aud = sm.result(rec), sm.audit(rec["stdout"])
        io, sched = snapped(rec["stdout"])
        print(f"== {name}: rc={rec['returncode']} {rec.get('gpu', '')}")
        print(f"   maxT={res['maxT']} oodN_maxT={res['oodN_maxT']} rungs={res['rungs']} ood={res['ood_rungs']}")
        print(f"   eval={res['eval']} train_s={res['train_s']} eval_s={res['eval_s']} state={res['state']}")
        print(f"   table rows right={aud.get('rows_right')} wrong={aud.get('wrong_rows')} {aud.get('audit_last')}")
        for l in sched:
            print("  ", l[:300])
        print("   snapped I/O:", {k: v for k, v in io.items() if k not in ("code", "wcode")})
        print("   code:", io.get("code"), " writer code:", io.get("wcode"))
        if all(k in io for k in ("N", "x", "T", "writer", "code", "wcode", "digits")):
            for split in ("depth_t_1", "depth_ood_n_t_1", "depth_ood_n_t_64"):
                print(f"   offline read of {split}:", check_reader(io, split))


if __name__ == "__main__":
    main(sys.argv[1:])
