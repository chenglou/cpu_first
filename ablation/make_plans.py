"""Write the ablation plans (plans/*.json) for launch.py. Every job logs the snapped table every 50 steps.

    python make_plans.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [74, 1, 2, 3]
LOG = {"TABLE_EVERY": "50"}

# Leave one out on the core recipe (submission_adder.py): each entry turns one ingredient off. Longest-running first.
LOO = {
    "noWarm": {"ADDER_WARM": "0"},                      # the designed adder for the first 70 steps
    "noWrap": {"ANCHOR_WRAP": "0.0"},                   # (x - N) - N does not borrow
    "noFreeze": {"FREEZE_WIN": "1000000000"},           # learning rate 0 after 200 exact steps
    "noLearnedCheck": {"ID_AFTER_WARM": "0.0"},         # the learned-adder check (D - N) - (0 - N) = D after the warm start
    "noIdBorrow": {"ID_BORROW": "0.0"},                 # the check's final borrow = the adder's final carry
    "noPlace0": {"ID_PLACE0": "1.0"},                   # weight 5 on the lowest place of the check
    "noLess": {"ANCHOR_LESS": "0.0"},                   # x - N borrows
    "noMore": {"ANCHOR_MORE": "0.0"},                   # N - x does not borrow
    "noNeg": {"ANCHOR_NEG": "0.0"},                     # 0 - N borrows
    "noXmod": {"SELF_LESSON": "False"},                 # x mod N = x (phase B)
    "noZero": {"ANCHOR_ZERO": "0.0"},                   # R - 0 = R
    "noSelf": {"ANCHOR_SELF": "0.0"},                   # R - R = 0
    "noNeg2": {"ANCHOR_NEG2": "0.0"},                   # 0 + N = N
    "noComm": {"ANCHOR_COMM": "0.0"},                   # R + N = N + R
    "softAnswers": {"ANSWER_PATH": "\"soft\""},         # the answers along the own hard path
    "noPhaseA": {"ID_STEPS": "1"},                      # phase B's lessons and the freeze wait for 80% of the checks
    "noAdamReset": {"ADAM_RESET": "False"},             # Adam's state cleared at the switch
    "liveLinks": {"ID_LINKS": "\"live\""},              # the check reads every link (also those after the first borrow)
    "ceAll": {"CE_SCOPE": "\"all\""},                   # remainder lessons score only the places below N's width
    "noHardAnchors": {"HARD_ANCHORS": "False"},         # the schedule waits for every lesson's hard check
}


# Add one in: the bare minimum is the check (the designed adder for the first 70 steps, then the learned adder; digits
# only, every place weighted the same) with the schedule (phase switch, Adam reset, freeze) and nothing else.
BARE = {"ID_BORROW": "0.0", "ID_PLACE0": "1.0", "ANCHOR_ZERO": "0.0", "ANCHOR_SELF": "0.0", "ANCHOR_LESS": "0.0",
        "ANCHOR_MORE": "0.0", "ANCHOR_WRAP": "0.0", "ANCHOR_NEG": "0.0", "ANCHOR_NEG2": "0.0", "ANCHOR_COMM": "0.0",
        "SELF_LESSON": "False", "REAL_MAX_T": "0"}
ADD = {"bare": {}, "Borrow": {"ID_BORROW": "1.0"}, "Place0": {"ID_PLACE0": "5.0"}, "Zero": {"ANCHOR_ZERO": "1.0"},
       "Self": {"ANCHOR_SELF": "1.0"}, "Less": {"ANCHOR_LESS": "1.0"}, "More": {"ANCHOR_MORE": "1.0"},
       "Wrap": {"ANCHOR_WRAP": "1.0"}, "Neg": {"ANCHOR_NEG": "1.0"}, "Neg2": {"ANCHOR_NEG2": "1.0"},
       "Comm": {"ANCHOR_COMM": "1.0"}, "Xmod": {"SELF_LESSON": "True"}, "Answers": {"REAL_MAX_T": "2"}}


def jobs(prefix: str, variants: dict, seeds: list) -> list:
    return [{"name": f"{prefix}_{v}_s{s}", "set": dict(LOG, **sets), "seed": s}
            for v, sets in variants.items() for s in seeds]


def write(name: str, js: list) -> None:
    (HERE / "plans").mkdir(exist_ok=True)
    (HERE / "plans" / f"{name}.json").write_text(json.dumps(js, indent=1))
    print(name, len(js), "jobs")


if __name__ == "__main__":
    # wave 1: leave one out (the rows' answers, REAL_MAX_T = 0, reuse the certified file's adder_warm_idonly_cpu_seed*
    # records: the same source at its defaults), plus the baseline on two seeds (reproduces adder_warm_cpu_seed74/1)
    write("w1_loo", jobs("loo", LOO, SEEDS) + jobs("loo", {"base": {}}, [74, 1]))
    # wave 2: add one in from the bare minimum, two seeds
    write("w2_aoi", jobs("aoi", {k: dict(BARE, **v) for k, v in ADD.items()}, [74, 1]))
    # wave 3: pairs that cover for each other (leave two out), and small recipes built up from the bare minimum
    on = lambda *names: {k: v for n in names for k, v in ADD[n].items()}
    nocheck = {"ID_AFTER_WARM": "0.0", "ID_GATE": "True"}   # no check after the warm start; its hard version gates
    pairs = {"noLessXmod": dict(LOO["noLess"], **LOO["noXmod"]),
             "noLearnedAdder": dict(nocheck, **LOO["noNeg2"], **LOO["noComm"])}
    small = {"LW": on("Less", "Wrap"), "LMW": on("Less", "More", "Wrap"), "LMWX": on("Less", "More", "Wrap", "Xmod"),
             "LMWZS": on("Less", "More", "Wrap", "Zero", "Self"),
             "factsLMW": dict(nocheck, **on("Less", "More", "Wrap")),
             "factsLMWA": dict(nocheck, **on("Less", "More", "Wrap", "Answers")),
             "factsLMWXA": dict(nocheck, **on("Less", "More", "Wrap", "Xmod", "Answers")),
             "factsLMWC": dict(nocheck, **on("Less", "More", "Wrap", "Comm"))}
    write("w3_pairs_small", jobs("pair", pairs, SEEDS) + jobs("aoi", {k: dict(BARE, **v) for k, v in small.items()},
                                                                [74, 1]))
    # wave 4: general rules in place of the hand-picked facts (submission_ablate.py ANCHOR_ORDER, ANCHOR_CHAIN)
    hand = {"ANCHOR_LESS": "0.0", "ANCHOR_MORE": "0.0", "ANCHOR_NEG": "0.0", "ANCHOR_ZERO": "0.0", "ANCHOR_SELF": "0.0"}
    order, chain = {"ANCHOR_ORDER": "1.0"}, {"ANCHOR_CHAIN": "1.0", "ANCHOR_WRAP": "0.0"}
    minimal = {"ANCHOR_NEG2": "0.0", "ANCHOR_COMM": "0.0", "SELF_LESSON": "False", "REAL_MAX_T": "0", "ID_BORROW": "0.0",
               "ID_PLACE0": "1.0"}
    write("w4_general", jobs("gen", {"noHand": hand, "order": dict(hand, **order), "chain": chain,
                                     "orderChain": dict(hand, **order, **chain),
                                     "minOrderChain": dict(hand, **order, **chain, **minimal)}, SEEDS))
    # wave 5: R - 0 = R and R - R = 0 leak through soft borrows into (r, 0, 1) and (r, r, 1), the two trap families
    # that the wrap-around fact and x mod N = x repair: without the two anchors, are the repairs still needed?
    zs = dict(LOO["noZero"], **LOO["noSelf"])
    write("w5_zeroself", jobs("pair", {"noZeroSelf": zs, "noZeroSelfWrap": dict(zs, **LOO["noWrap"]),
                                       "noZeroSelfWrapXmod": dict(zs, **LOO["noWrap"], **LOO["noXmod"])}, SEEDS))
    # wave 6: the general rule "train only the rows a lesson uses": the two anchors ripple with the hard borrow-in
    hc = {"HARD_CIN": "True"}
    write("w6_hardcin", jobs("gen", {"hardCin": hc, "hardCinNoWrap": dict(hc, **LOO["noWrap"])}, SEEDS))
    # wave 7: the candidate simplified recipe on 8 seeds: the bare check + x - N borrows + N - x does not + the wrap fact,
    # with the schedule as it is (phase switch, Adam reset, freeze) and with no schedule at all
    simple = dict(BARE, **on("Less", "More", "Wrap"))
    nosched = {"ID_STEPS": "1", "ADAM_RESET": "False", "FREEZE_WIN": "1000000000"}
    write("w7_simple", jobs("simple", {"LMW": simple, "LMWnosched": dict(simple, **nosched)}, [74, 1, 2, 3, 4, 5, 6, 7]))
    # wave 8: the facts alone after the warm start (no check lesson, no learned adder in training), no schedule
    facts = dict(BARE, **on("Less", "More", "Wrap"), **{"ID_AFTER_WARM": "0.0"})
    write("w8_facts", jobs("simple", {"factsLMWnosched": dict(facts, **nosched)}, [74, 1, 2, 3, 4, 5, 6, 7]))
    # wave 9: without R - 0 = R and R - R = 0 there is no trap to repair: is the wrap-around fact still needed?
    write("w9_nowrap", jobs("aoi", {"LM": dict(BARE, **on("Less", "More")),
                                    "factsLM": dict(BARE, **on("Less", "More"), **nocheck)}, SEEDS))
    # wave C1: the decisive ablations repeated on the certified combined model (combined/submission_combined_ablate.py)
    cfile = "combined/submission_combined_ablate.py"
    cz = {"ANCHOR_ZERO": "0.0", "ANCHOR_SELF": "0.0"}
    ctable = dict(cz, **{"ANCHOR_NEG": "0.0", "ANCHOR_NEG2": "0.0", "ANCHOR_COMM": "0.0", "ID_BORROW": "0.0",
                         "ID_PLACE0": "1.0"})
    cjobs = jobs("cmb", {"noZeroSelf": cz, "noWrap": {"ANCHOR_WRAP": "0.0"}, "simpleTable": ctable}, SEEDS)
    cjobs += jobs("cmb", {"noLearnedCheck": {"ID_AFTER_WARM": "0.0"},
                          "noIdBorrowPlace0": {"ID_BORROW": "0.0", "ID_PLACE0": "1.0"},
                          "noZeroSelfWrap": dict(cz, **{"ANCHOR_WRAP": "0.0"})}, [74, 1])
    for j in cjobs:
        j["file"] = cfile
    write("c1_combined", cjobs)
    # wave 10: with the trap-making anchors gone, how short can the hand-written adder's warm start be?
    write("w10_warm", jobs("simple", {f"LMWwarm{w}": dict(simple, ADDER_WARM=str(w)) for w in (0, 10, 20, 30, 50)},
                           [74, 1]))
    # wave 11: leave one out on the simplified recipe itself (LMW: the check + x - N, N - x, wrap): what is left of it
    write("w11_simple_loo", jobs("aoi", {"MW": dict(BARE, **on("More", "Wrap"))}, SEEDS)
          + jobs("aoi", {"LW": dict(BARE, **on("Less", "Wrap")),
                         "factsLMW": dict(BARE, **on("Less", "More", "Wrap"), **nocheck)}, [2, 3]))
    # wave 12: a simplified recipe with no register-width fact: the bare check + x - N, N - x + R - 0 = R, R - R = 0
    # rippled with the hard borrows (Rule 3), without the wrap-around fact; with the schedule and without
    general = dict(BARE, **on("Less", "More", "Zero", "Self"), HARD_CIN="True")
    write("w12_general_simple", jobs("simple", {"LMZShc": general, "LMZShcNosched": dict(general, **nosched)}, SEEDS))
    # wave 3b: does the wrap-around fact matter only against the learned check's barrier at row (0, 0, 1)?
    write("w3b_wrap", jobs("pair", {"noCheckWrap": dict(nocheck, **LOO["noWrap"]),
                                    "noCheckWrapComm": dict(nocheck, **LOO["noWrap"], **LOO["noComm"])}, SEEDS))
