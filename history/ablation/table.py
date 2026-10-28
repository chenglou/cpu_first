"""ABLATION.md's ingredient table from the run records (python table.py). Each ingredient: what it is, why it was added
(the write-up that added it), its leave-one-out and add-one-in runs, and whether it assumes today's hand-made wiring."""

import sys

from analyze import summary

SEEDS = [74, 1, 2, 3]
loo = lambda v: [f"loo_{v}_s{s}" for s in SEEDS]
aoi = lambda v: [f"aoi_{v}_s{s}" for s in (74, 1)]
idonly = [f"adder_warm_idonly_cpu_seed{s}" for s in SEEDS]

# (ingredient, constant, why added, leave-one-out records, add-one-in records or a note, wiring)
ROWS = [
    ("warm start: the hand-written adder in the check for steps 1–70", "`ADDER_WARM = 70`",
     "ADDER.md §5.2: from a random start the learned-adder check settles on the lazy table (46 runs, 0 certified)",
     loo("noWarm"), "in the bare minimum", "general: checks the cell on the operands it is given"),
    ("the learned-adder check (D − N) − (0 − N) = D after the warm start", "`ID_AFTER_WARM` (new)",
     "ADDER.md: the point of the piece, no hand-written arithmetic after the warm start",
     loo("noLearnedCheck"), "in the bare minimum", "tied: the adder A − (0 − B) and the chain's links"),
    ("complete identity: the check's final borrow = the adder's final carry", "`ID_BORROW = 1`",
     "CHAIN.md §5.3: the digits-only check has a local minimum at (0, 0, 1); seeds stalled 350–1,150 steps, one failed",
     loo("noIdBorrow"), aoi("Borrow"), "tied: the stop rule reads the final borrow"),
    ("lowest place weighted 5×", "`ID_PLACE0 = 5`",
     "LEARNED_REMAINDER.md §3.4, §4.2: the swapped borrow code differs only at place 0; two first official runs fell there",
     loo("noPlace0"), aoi("Place0"), "tied: place 0 is the ripple's start (the fixed borrow-in)"),
    ("R − 0 = R on the registers", "`ANCHOR_ZERO`", "ADDER.md §1.2: the plan's lessons (pins the column m = 0)",
     loo("noZero"), aoi("Zero"), "general (any numbers)"),
    ("R − R = 0 on the registers", "`ANCHOR_SELF`", "ADDER.md §1.2: the plan's lessons (pins the diagonal)",
     loo("noSelf"), aoi("Self"), "general (any numbers)"),
    ("x − N borrows (x < N)", "`ANCHOR_LESS`",
     "ADDER.md §4: with N − x, teaches all 200 borrows; replaces the designed carry's comparison",
     loo("noLess"), aoi("Less"), "general fact (needs the reader's x and N)"),
    ("N − x does not borrow (x < N)", "`ANCHOR_MORE`", "ADDER.md §4: as above", loo("noMore"), aoi("More"),
     "general fact (needs the reader's x and N)"),
    ("(x − N) − N does not borrow (a wrapped x − N is large)", "`ANCHOR_WRAP`",
     "ADDER.md §5.1: the (r, 0, 1) trap ten entries from the truth; without it the warm start stalls at 168–170",
     loo("noWrap"), aoi("Wrap"), "tied: 9 register places for 8-digit moduli"),
    ("0 − N borrows (N > 0), after the warm start", "`ANCHOR_NEG`",
     "ADDER.md §2: makes the learned adder's carry 1 − b_f; the plan's lessons", loo("noNeg"), aoi("Neg"),
     "general fact"),
    ("0 + N = N with the learned adder, after the warm start", "`ANCHOR_NEG2`",
     "ADDER.md §4: knocks out the lazy table from a random start", loo("noNeg2"), aoi("Neg2"),
     "tied: the adder A − (0 − B)"),
    ("R + N = N + R with the learned adder, after the warm start", "`ANCHOR_COMM`",
     "ADDER.md §4: knocks out the lazy table from a random start", loo("noComm"), aoi("Comm"),
     "tied: the adder A − (0 − B)"),
    ("x mod N = x (phase B, soft own-borrow choice over one division step)", "`SELF_LESSON`",
     "LEARNED_REMAINDER.md §4.2: fixes the identity's (r, r, 1) saddle; CHAIN.md §5.4: still helps",
     loo("noXmod"), aoi("Xmod"), "tied: the chain and the stop rule (its content is x − N borrows)"),
    ("the rows' answers, T ≤ 2 (phase B)", "`REAL_MAX_T = 2`",
     "LEARNED_REMAINDER.md §4.2: a direct lesson on stall rows such as (9, 9, 0)", idonly, aoi("Answers"),
     "general (labels)"),
    ("the answers along the machine's own hard path", "`ANSWER_PATH = \"hard\"`",
     "LEARNED_REMAINDER.md §3.3: soft choices teach biased lessons; 10× cheaper", loo("softAnswers"),
     "only with the answers", "general"),
    ("phase switch: phase B's lessons and the freeze wait for 80% of the checks (or 300 steps)",
     "`ID_SWITCH`, `STAGE_WIN`, `ID_STEPS`",
     "LEARNED_REMAINDER.md §4.2: x mod N = x from the start biases every borrow toward 1", loo("noPhaseA"),
     "in the bare minimum", "general"),
    ("Adam's state cleared at the switch", "`ADAM_RESET` (new)",
     "LEARNED_REMAINDER.md §3.3: phase A's large gradients in Adam's memory slowed the repairs", loo("noAdamReset"),
     "in the bare minimum", "general"),
    ("freeze: learning rate 0 after 200 exact steps", "`FREEZE_WIN = 200`",
     "LEARNED_REMAINDER.md §3.1 (stops training once the self-checks are exact)", loo("noFreeze"),
     "in the bare minimum", "general"),
    ("the check reads every link, also after the chain's first borrow", "`ID_LINKS` (new)",
     "CHAIN.md §4: still valid checks; §5.4: live links only did not fix the stall", loo("liveLinks"),
     "in the bare minimum", "tied: the chain"),
    ("the remainder lessons score only the places below N's width", "`CE_SCOPE` (new)",
     "LEARNED_REMAINDER.md §3.2: over-subtracting trials write 9s above N's width", loo("ceAll"),
     "only with x mod N = x or the answers", "tied: the layout"),
    ("the freeze and the switch read every lesson's hard check", "`HARD_ANCHORS` (new)",
     "ADDER.md §2: each anchor's hard version joins the checks the schedule reads", loo("noHardAnchors"),
     "only with anchors", "general"),
]


S8 = [74, 1, 2, 3, 4, 5, 6, 7]
names = lambda prefix, v, seeds: [f"{prefix}_{v}_s{s}" for s in seeds]

# (what, records) per table
SMALL = [
    ("bare + x − N borrows + wrap", names("aoi", "LW", [74, 1, 2, 3])),
    ("bare + N − x no borrow + wrap", names("aoi", "MW", SEEDS)),
    ("bare + x − N + N − x (no wrap)", names("aoi", "LM", SEEDS)),
    ("**bare + x − N + N − x + wrap (the simplified recipe)**", names("aoi", "LMW", [74, 1])),
    ("the same, on 8 seeds", names("simple", "LMW", S8)),
    ("the same, no schedule (no phase switch, no Adam reset, no freeze): `submission_simple.py`'s training",
     names("simple", "LMWnosched", S8)),
    ("+ x mod N = x", names("aoi", "LMWX", [74, 1])),
    ("+ R − 0 = R, R − R = 0", names("aoi", "LMWZS", [74, 1])),
    ("x − N + N − x + wrap, no check lesson after the warm start (the learned check only gates the freeze)",
     names("aoi", "factsLMW", [74, 1, 2, 3])),
    ("the same, no schedule", names("simple", "factsLMWnosched", S8)),
    ("the same + the answers", names("aoi", "factsLMWA", [74, 1])),
    ("the same + x mod N = x + the answers", names("aoi", "factsLMWXA", [74, 1])),
    ("the same + R + N = N + R", names("aoi", "factsLMWC", [74, 1])),
    ("x − N + N − x, no check lesson, no wrap", names("aoi", "factsLM", SEEDS)),
]
PAIRS = [
    ("x − N borrows, x mod N = x", names("pair", "noLessXmod", SEEDS)),
    ("the learned adder's lessons after the warm start: its check, 0 + N = N, R + N = N + R", names("pair", "noLearnedAdder", SEEDS)),
    ("R − 0 = R, R − R = 0", names("pair", "noZeroSelf", SEEDS)),
    ("R − 0 = R, R − R = 0, wrap", names("pair", "noZeroSelfWrap", SEEDS)),
    ("R − 0 = R, R − R = 0, wrap, x mod N = x", names("pair", "noZeroSelfWrapXmod", SEEDS)),
    ("the learned check, wrap", names("pair", "noCheckWrap", SEEDS)),
    ("the learned check, wrap, R + N = N + R", names("pair", "noCheckWrapComm", SEEDS)),
]
GENERAL = [
    ("x − N, N − x, 0 − N, R − 0 = R, R − R = 0 all off", names("gen", "noHand", SEEDS)),
    ("… replaced by the order rule", names("gen", "order", SEEDS)),
    ("wrap replaced by the chain rule", names("gen", "chain", SEEDS)),
    ("both replacements", names("gen", "orderChain", SEEDS)),
    ("both, and nothing else after the warm start but the check", names("gen", "minOrderChain", SEEDS)),
    ("R − 0 = R, R − R = 0 rippled with the hard borrows", names("gen", "hardCin", SEEDS)),
    ("the same, without wrap", names("gen", "hardCinNoWrap", SEEDS)),
    ("the bare check + x − N + N − x + the two anchors rippled with the hard borrows, no wrap",
     names("simple", "LMZShc", SEEDS)),
    ("the same, no schedule", names("simple", "LMZShcNosched", SEEDS)),
]
WARM = [(f"hand-written adder for the first {w} steps", names("simple", f"LMWwarm{w}", [74, 1]))
        for w in (0, 10, 20, 30, 50)] + [("70 steps (the recipe)", names("aoi", "LMW", [74, 1]))]
COMBINED = [
    ("R − 0 = R, R − R = 0 off", names("cmb", "noZeroSelf", SEEDS)),
    ("wrap off", names("cmb", "noWrap", SEEDS)),
    ("the simplified table recipe (also 0 − N, 0 + N = N, R + N = N + R off; digits-only check, places alike)",
     names("cmb", "simpleTable", SEEDS)),
    ("the learned check off after the warm start (its hard version still gates)", names("cmb", "noLearnedCheck", [74, 1])),
    ("complete identity and lowest-place weight off", names("cmb", "noIdBorrowPlace0", [74, 1])),
    ("R − 0 = R, R − R = 0, wrap off", names("cmb", "noZeroSelfWrap", [74, 1])),
]


def rows_table(title, rows):
    print(f"| {title} | certified, frozen at, failures |")
    print("|---|---|")
    for what, recs in rows:
        print(f"| {what} | {summary(recs)} |")
    print()


LOO_LABELS = {
    "base": "nothing (the ablation copy at its defaults)", "noWarm": "warm start", "noLearnedCheck": "learned-adder check",
    "noIdBorrow": "complete identity", "noPlace0": "lowest place ×5", "noZero": "R − 0 = R", "noSelf": "R − R = 0",
    "noLess": "x − N borrows", "noMore": "N − x no borrow", "noWrap": "(x − N) − N no borrow", "noNeg": "0 − N borrows",
    "noNeg2": "0 + N = N", "noComm": "R + N = N + R", "noXmod": "x mod N = x", "softAnswers": "own hard path for the answers",
    "noPhaseA": "phase switch", "noAdamReset": "Adam reset", "noFreeze": "freeze", "liveLinks": "every link checked",
    "ceAll": "score below N's width", "noHardAnchors": "freeze reads every check"}


def loo_table() -> None:
    import contextlib, io
    from analyze import variants
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        variants(["loo"])
    lines = buf.getvalue().rstrip().splitlines()
    print("| removed | " + lines[0].lstrip("| "))
    print("|---" + lines[1])
    order = list(LOO_LABELS)
    body = sorted(lines[2:], key=lambda l: order.index(l.split("|")[1].strip()))
    for l in body:
        v = l.split("|")[1].strip()
        print(f"| {LOO_LABELS[v]} " + l)
    # the answers: the certified file's own runs that never see one (REAL_MAX_T = 0, the same source)
    import json
    from analyze import parse, find
    infos = [parse(json.loads(find(n).read_text())) for n in idonly]
    j = lambda k: ", ".join(str(i[k] or "–") for i in infos)
    print(f"| the answers (`adder_warm_idonly_cpu_seed*`) | noAnswers | {j('seed')} | "
          f"{sum(i['certified'] for i in infos)}/{len(infos)} | {j('switch')} | {j('frozen')} | {j('right')} |  |")


def fill(doc: str) -> None:
    """Replace each <!-- TABLE:name --> ... <!-- /TABLE --> block of the doc with the table from the records."""
    import contextlib, io, re
    from analyze import variants
    makers = {"loo": loo_table,
              "ingredients": lambda: main_tables(ingredients=True, others=False),
              "small": lambda: rows_table("recipe", SMALL), "pairs": lambda: rows_table("removed together", PAIRS),
              "general": lambda: rows_table("general replacement", GENERAL),
              "warm": lambda: rows_table("warm start", WARM),
              "combined": lambda: rows_table("combined model: removed", COMBINED)}
    text = open(doc).read()
    for name, make in makers.items():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            make()
        text = re.sub(rf"<!-- TABLE:{name} -->.*?<!-- /TABLE -->",
                      lambda m: f"<!-- TABLE:{name} -->\n{buf.getvalue().rstrip()}\n<!-- /TABLE -->", text, flags=re.S)
    open(doc, "w").write(text)


def main_tables(ingredients=True, others=True):
    if ingredients:
        print("| ingredient | constant | why it was added | leave one out (seeds 74, 1, 2, 3) | add one in (bare + it; "
              "seeds 74, 1) | wiring |")
        print("|---|---|---|---|---|---|")
        for name, const, why, l, a, wiring in ROWS:
            a_cell = summary(a) if isinstance(a, list) else a
            print(f"| {name} | {const} | {why} | {summary(l)} | {a_cell} | {wiring} |")
        print()
    if others:
        rows_table("recipe", SMALL)
        rows_table("removed together", PAIRS)
        rows_table("general replacement", GENERAL)
        rows_table("warm start", WARM)
        rows_table("combined model: removed", COMBINED)


def main():
    if sys.argv[1:2] == ["fill"]:
        fill(sys.argv[2])
    else:
        main_tables(ingredients="others" not in sys.argv[1:], others="ingredients" not in sys.argv[1:])


if __name__ == "__main__":
    main()
