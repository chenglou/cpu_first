"""Checks official records against the files they certify (REVIEW.md). Standard library only; reads, never writes.

    python3 tools/verify_records.py                        # the file to judge's 8 runs, then the wider file's 8,
                                                           # the family file's 8, the legal file's 8, fast2's 8,
                                                           # s3i's 8, stage 1 and general v3
    python3 tools/verify_records.py REC.json[=FILE.py] ... # any records; FILE defaults to the record's submission_file
    python3 tools/verify_records.py --log RUN.log [REC]    # a run of your own: its RESULT_JSON line against REC.json
                                                           # (default: the file to judge's official record)
    python3 tools/verify_records.py --data DIR             # an H1 folder's md5, computed as tools/modal_runner.py does

For each record: the md5 of the source it embeds against the file's md5 now, its manifest against the competition's
benchmark/manifests/h100_hard_h1.json (../one-layer-deeper), the evaluator's RESULT_JSON line in its stdout against
its result, the certified T on both ladders, every rung's count, the three splits, and the run's times. A path the
early work had before it moved into history/ (runs/..., tournament/...) is looked up there.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent                       # cpu_first/
COMP = HERE.parent / "one-layer-deeper"                              # the competition clone at 87e5c26
MANIFEST = COMP / "benchmark" / "manifests" / "h100_hard_h1.json"
DATA_MD5 = "a6414ec9fc6a5efce6414895b52d4eb7"                        # runner.data_md5 of every shared-runner H1 record
JUDGE = "integration/legal_deep/submission_stage3_deep.py"             # the file to judge (REVIEW.md §2) ...
JUDGE_RECORDS = "integration/legal_deep/records/legal_deep"            # ... and its records: _official, _h100_seed<N>
WIDER = "integration/legal_wider/submission_stage3_wider.py"           # the file to judge before it (REVIEW.md §2) ...
WIDER_RECORDS = "integration/legal_wider/records/legal_wider"          # ... and its records, named the same way
FAMILY = "integration/legal_family/submission_stage3_family.py"        # the file to judge before that ...
FAMILY_RECORDS = "integration/legal_family/records/legal_family"       # ... and its records, named the same way
LEGAL = "integration/combined_legal/submission_stage3_legal.py"        # and before that ...
LEGAL_RECORDS = "integration/combined_legal/records/combined_legal"    # ... and its records, named the same way
FAST2 = "integration/combined_fast/submission_stage3_fast2.py"         # and before the legal file ...
FAST2_RECORDS = "integration/combined_fast/records/combined_fast2"     # ... and its records, named the same way
FRONTIER = [(f"{JUDGE_RECORDS}_official.json", JUDGE)]
FRONTIER += [(f"{JUDGE_RECORDS}_h100_seed{s}.json", JUDGE) for s in range(1, 8)]
FRONTIER += [(f"{WIDER_RECORDS}_official.json", WIDER)]
FRONTIER += [(f"{WIDER_RECORDS}_h100_seed{s}.json", WIDER) for s in range(1, 8)]
FRONTIER += [(f"{FAMILY_RECORDS}_official.json", FAMILY)]
FRONTIER += [(f"{FAMILY_RECORDS}_h100_seed{s}.json", FAMILY) for s in range(1, 8)]
FRONTIER += [(f"{LEGAL_RECORDS}_official.json", LEGAL)]
FRONTIER += [(f"{LEGAL_RECORDS}_h100_seed{s}.json", LEGAL) for s in range(1, 8)]
FRONTIER += [(f"{FAST2_RECORDS}_official.json", FAST2)]
FRONTIER += [(f"{FAST2_RECORDS}_h100_seed{s}.json", FAST2) for s in range(1, 8)]
FRONTIER += [("integration/records/s3i_official.json", "integration/submission_stage3_s3i.py"),
             ("integration/records/s1_official.json", "integration/submission_stage1.py"),
             ("general_lessons/fixed/records/v3k_h1_official.json", "general_lessons/submission_general_v3.py")]
FRONTIER += [(f"integration/records/s3i_h100_seed{s}.json", "integration/submission_stage3_s3i.py") for s in range(1, 8)]


def at(p: str | None) -> str | None:
    """p as given, or under history/ when it names an early file by its place before the move."""
    return f"history/{p}" if p and not (HERE / p).exists() and (HERE / "history" / p).exists() else p


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def data_md5(root: Path) -> str:
    h = hashlib.md5()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def result_json(stdout: str) -> dict | None:
    lines = [l for l in stdout.splitlines() if l.startswith("RESULT_JSON=")]
    return json.loads(lines[-1][len("RESULT_JSON="):]) if lines else None


def ladders(result: dict) -> list[str]:
    """What certification rests on, one line per seed: T on both ladders, every rung's count, the splits."""
    out = []
    for s in result["seeds"]:
        dp = s["depth_profile"]
        rungs = " ".join(f'{r["correct_examples"]}/{r["example_count"]}' for r in dp["rungs"])
        ood = " ".join(f'{r["correct_examples"]}/{r["example_count"]}' for r in dp["ood_n_rungs"])
        splits = ", ".join(f'{k} {v["correct_examples"]}/{v["example_count"]}' for k, v in sorted(s["evaluation"].items()))
        out.append(f'seed {s["seed"]}: T = {dp["max_certified_time_steps"]} / {dp["ood_n_max_certified_time_steps"]}; '
                   f'pool rungs {rungs}; unseen-N rungs {ood}; {splits}')
    return out


def manifest_diff(m: dict | str) -> dict:
    """{key: (official, record's)}; the older runners recorded the official manifest by its path instead."""
    if isinstance(m, str):
        return {} if m == "benchmark/manifests/h100_hard_h1.json" else {"manifest": ("the official file", m)}
    official = json.loads(MANIFEST.read_text())
    return {f"{sec}.{k}": (official[sec].get(k), m.get(sec, {}).get(k)) for sec in official
            if isinstance(official[sec], dict) for k in set(official[sec]) | set(m.get(sec, {}))
            if official[sec].get(k) != m.get(sec, {}).get(k)} | \
           {sec: (official[sec], m.get(sec)) for sec in official if not isinstance(official[sec], dict)
            and official[sec] != m.get(sec)}


def check(rec_path: str, file_path: str | None) -> bool:
    rec_path = at(rec_path)
    r = json.loads((HERE / rec_path).read_text())
    file_path = at(file_path or r.get("submission_file"))
    emb = md5(r["submission"].encode("utf-8")) if "submission" in r else "none (the record embeds no source)"
    now = md5((HERE / file_path).read_bytes()) if file_path else None
    diff = manifest_diff(r.get("manifest") or "benchmark/manifests/h100_hard_h1.json")   # SpecBit's: no field
    seeds_only = set(diff) <= {"runtime.seeds"}
    rj = result_json(r.get("stdout") or r.get("log") or "")                            # the oldest keep "log"
    full = all(x["correct_examples"] == x["example_count"] for s in r["result"]["seeds"]
               for x in s["depth_profile"]["rungs"] + s["depth_profile"]["ood_n_rungs"] + list(s["evaluation"].values()))
    ok = emb == now and seeds_only and rj == r["result"] and full and r["returncode"] == 0
    print(f'{"OK " if ok else "BAD"} {rec_path}')
    verdict = "same" if emb == now else "DIFFERENT" if "submission" in r else "cannot be compared"
    print(f'    embedded source md5 {emb}; {file_path} now {now}: {verdict}')
    print(f'    manifest: {"the official file" if not diff else "official except " + json.dumps(diff)}'
          f'{"" if seeds_only else "  <- not only the seed"}; GPU {r.get("gpu")}; returncode {r["returncode"]}')
    print(f'    RESULT_JSON line in stdout equals result: {rj == r["result"]}')
    for line in ladders(r["result"]):
        print(f"    {line}")
    for s in r["result"]["seeds"]:
        print(f'    training {s["training_seconds"]:.1f} s ({s["completed_training_steps"]} steps), evaluation '
              f'{s["evaluation_seconds"]:.1f} s, {s["model_state_elements"]:,} parameters; data md5 '
              f'{(r.get("runner") or {}).get("data_md5", "not recorded")}')
    return ok


def compare_log(log: str, rec_path: str = FRONTIER[0][0]) -> bool:
    mine = result_json(Path(log).read_text())
    rec_path = at(rec_path)
    ref = json.loads((HERE / rec_path).read_text())["result"]
    if mine is None:
        print(f"no RESULT_JSON= line in {log}")
        return False
    same = ladders(mine) == ladders(ref) if len(mine["seeds"]) == len(ref["seeds"]) else False
    print(f'{"OK " if same else "BAD"} {log} against {rec_path}: certification fields {"equal" if same else "differ"}')
    for tag, res in (("yours", mine), ("record", ref)):
        for line in ladders(res):
            print(f"    {tag}: {line}")
        for s in res["seeds"]:
            print(f'    {tag}: training {s["training_seconds"]:.1f} s, evaluation {s["evaluation_seconds"]:.1f} s')
    return same


def main(argv: list[str]) -> int:
    if argv[:1] == ["--data"]:
        h = data_md5(Path(argv[1]))
        print(h, "(the records' data)" if h == DATA_MD5 else f"(not the records' {DATA_MD5})")
        return 0 if h == DATA_MD5 else 1
    if argv[:1] == ["--log"]:
        return 0 if compare_log(*argv[1:3]) else 1
    pairs = [tuple(a.split("=", 1)) if "=" in a else (a, None) for a in argv] or FRONTIER
    oks = [check(rec, f) for rec, f in pairs]
    print(f"{sum(oks)} of {len(oks)} records check out")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
