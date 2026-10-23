"""Tiny core for the code tournament (FIXED_MODULUS.md section 3): submission_general_v3.py's own census and scoring
(digit_facts, census, code_score, code_lesson) on a fixed-modulus dataset's real training batches, with the machine
exact (MODE = "designed": every table and switch at the truth), so that only the tournament is under test. The reader's
digit-order facts are gathered on MINE_WIN + 2 batches first (io_lessons, as in training), then each batch scores the
contestants:

  mode = hard   v3 as built: a (contestant, lesson set, round) triple that fails an answer row is out; the survivors'
                sets and round options train the selector and the round menu (Adam, the file's LR); stop when the file
                chooses its code (ctl.leader)
  mode = soft   the tournament of TOURNAMENT.md instead: one logit per contestant (Adam, LR_CODE = 0.5), loss
                -log sum of the passing triples' weights (contestant x set x round options, the latter two the file's
                selector and round menu), contestants more than PRUNE = 12 nats behind the best are no longer run;
                stop when one contestant is left

Both start the selector and the round menu from the file's random start, so the true code must be found jointly with
long division and the plain square among 2 x 40 (set, round) combinations.

    ~/github/sotaku/venv/bin/python general_lessons/fixed/tiny_tournament.py DATA [mode=hard] [seed=0] [steps=40]
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("GENERAL_FILE", os.path.join(os.path.dirname(HERE), "submission_general_v3.py"))
sys.path.insert(0, os.path.dirname(HERE))
import local_check as LC  # noqa: E402
import torch  # noqa: E402

LR_CODE, PRUNE = 0.5, 12.0


def main(argv):
    data = argv[0]
    opt = dict(a.split("=", 1) for a in argv[1:])
    mode, seed, steps = opt.get("mode", "hard"), int(opt.get("seed", 0)), int(opt.get("steps", 40))
    sc, m = LC.designed_model(data, [])
    torch.manual_seed(seed)
    with torch.no_grad():                                                   # the selector and round: a random start
        m.lsel.copy_(sc.INIT * torch.randn(len(sc.PREDICATES)))
        for k in sc.ROUND_SW:
            m.rnd[k].copy_(sc.INIT * torch.randn(len(sc.ROUND_MENU[k])))
    train = LC.rows(data, "train", 10 ** 9)
    rng = random.Random(seed)
    ctl = m.ctl
    ctl.io = 3
    ctl.wire_ios = [True] * len(sc.PREDICATES)
    params = [m.lsel] + list(m.rnd.values())
    adam = torch.optim.Adam(params, lr=sc.LR)
    cw = None
    t0 = time.time()
    out = {"data": os.path.basename(data), "mode": mode, "seed": seed, "steps": []}

    def batch():
        rs = rng.sample(train, min(512, len(train)))
        ids, am, lab, tpos = LC.batch(rs)
        e, end = m.ext(ids, am)
        return rs, ids, am, lab, tpos, e, end

    for _ in range(sc.MINE_WIN + 2):                                        # the reader's digit-order facts
        rs, ids, am, lab, tpos, e, end = batch()
        with torch.no_grad():
            k = m.selected(e, end)
            m.io_lessons(e, end, k, m.ranks(k), torch.zeros(len(rs), sc.W, dtype=torch.long))
    out["ambiguous"] = ctl.ambiguous
    out["readings"] = {"".join(map(str, sc.HYP[h])): ctl.dcount[h] for h in ctl.readings}
    print(f"facts: ambiguous={ctl.ambiguous} readings={out['readings']}", flush=True)
    if not ctl.ambiguous:
        print("RESULT " + json.dumps(out), flush=True)
        return out
    per = [{"div_ok": 1.0, "law_ok": 1.0, "step_ok": 1.0} for _ in sc.PREDICATES]
    true_h = sc.HIDX[(0, 1, 1, 1)]
    for step in range(1, steps + 1):
        rs, ids, am, lab, tpos, e, end = batch()
        ctl.step += 1
        aux = {"per": per, "table_exact": 1.0, "lad_ok": 1.0, "e": e, "end": end, "tabs": m.tables(),
               "k": m.selected(e, end)}
        valid = lab != -100
        t1 = time.time()
        runs0 = ctl.code_runs
        if mode == "hard":
            adam.zero_grad()
            loss = m.code_lesson(aux, lab, valid, tpos)
            loss.backward()
            adam.step()
            alive = ctl.alive or {}
            n_alive = sum(0 if t is None else int(t[:, 0].unique().numel()) for t in alive.values())
            rec = {"step": step, "contestants_alive": n_alive, "triples": sum(0 if t is None else t.shape[0]
                                                                              for t in alive.values()),
                   "machine_runs": ctl.code_runs - runs0, "s": round(time.time() - t1, 2)}
            done = ctl.leader is not None
        else:
            if ctl.cands is None:
                m.census(e.device)
                cw = torch.zeros(ctl.cands[0].numel(), requires_grad=True)
                cadam = torch.optim.Adam([cw], lr=LR_CODE)
            with torch.no_grad():
                active = torch.nonzero(cw >= cw.max() - PRUNE).flatten()
                b = m.code_rows(aux, lab, valid, tpos)
                tri = []
                for h in ctl.cands[0][active].unique().tolist():
                    rt = 3 - sc.HYP[h][0] - sc.HYP[h][2]
                    rows = torch.nonzero(b["ok"] & (b["length"][:, rt] == 1) & (b["A"][:, 0] >= 0)).flatten()
                    pres = torch.zeros(rows.numel(), 10, dtype=torch.bool)
                    rn, dn, rx, dx = sc.HYP[h]
                    for D in (b["R"][rows, rn, dn], b["R"][rows, rx, dx], b["R"][rows, rt, 0, :sc.K]):
                        pres |= (torch.nn.functional.one_hot(D.clamp(min=0), 10).bool() & (D >= 0)[..., None]).any(1)
                    rows = rows[pres.sum(1).argsort(stable=True)]
                    cid = active[ctl.cands[0][active] == h]
                    tri.append(m.code_score(h, None, cid, rows, b, list(range(len(sc.PREDICATES))), aux["tabs"]))
                tri = torch.cat(tri)
            adam.zero_grad()
            cadam.zero_grad()
            if tri.shape[0]:
                lwc = cw[active].log_softmax(0)
                pos = torch.searchsorted(active, tri[:, 0])
                lw = lwc[pos] + m.lsel.log_softmax(-1)[tri[:, 1]] + m.round_logw(
                    torch.tensor(sc.ROUND_COMBOS)[tri[:, 2]])
                loss = -torch.logsumexp(lw, 0)
                loss.backward()
                adam.step()
                cadam.step()
            n_act = int((cw >= cw.max() - PRUNE).sum())
            rec = {"step": step, "contestants_active": n_act, "passing_triples": int(tri.shape[0]),
                   "machine_runs": ctl.code_runs - runs0, "s": round(time.time() - t1, 2)}
            done = n_act == 1
            if done:
                lead = int(cw.argmax())
                ctl.leader = (int(ctl.cands[0][lead]), ctl.cands[1][lead])
        rec["selector"] = [round(float(p), 3) for p in m.lsel.detach().softmax(-1)]
        if mode == "hard" and ctl.alive and all(t is not None for t in ctl.alive.values()):
            tri = torch.cat(list(ctl.alive.values()))
            pairs = torch.unique(tri[:, 1:], dim=0)[:8].tolist()
            rec["sets_rounds_alive"] = [(sc.PREDICATES[s_], [sc.ROUND_MENU[kk][i] for kk, i in
                                         zip(sc.ROUND_SW, sc.ROUND_COMBOS[c_])]) for s_, c_ in pairs]
        rec["round"] = m.round_names()
        out["steps"].append(rec)
        print(json.dumps(rec), flush=True)
        if done:
            break
    if ctl.leader is not None:
        h, lab_ = ctl.leader
        code = (lab_[ctl.dtok] if lab_.numel() > 10 else lab_).tolist()
        out.update(leader_reading="".join(map(str, sc.HYP[h])), leader_code=code,
                   leader_true=(h == true_h and code == list(range(10))), steps_to_choose=out["steps"][-1]["step"])
    out.update(census=int(ctl.cands[0].numel()) if ctl.cands is not None else None, seconds=round(time.time() - t0, 1),
               machine_runs=ctl.code_runs, history=ctl.code_hist, set=sc.PREDICATES[int(m.lsel.argmax())],
               round=m.round_names())
    print("RESULT " + json.dumps(out), flush=True)
    if opt.get("out"):
        with open(opt["out"], "w") as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
