"""The leave-one-out arms of sp2's law book and choosing rule (integration/sp_ablation/SP_ABLATION.md): each variant is
sp2 (integration/self_practice/submission_stage3_sp2.py, md5 afc70fe9) with one part removed or swapped by a few text
edits, each asserted to match exactly once. The rules sp2 already has (lp, ct2a2, all, lpu, lpc) run sp2 itself
through the runner's --set. Writes variants/<arm>.py, variants/MD5SUMS.json and plans/arms.json (every arm's file and
--set); reads sp2 and never writes it.

    ~/github/sotaku/venv/bin/python integration/sp_ablation/make_variants.py            # build, or check a build
    ~/github/sotaku/venv/bin/python integration/sp_ablation/make_variants.py --diff ARM   # one arm's edits as a diff
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP2 = HERE.parent / "self_practice" / "submission_stage3_sp2.py"
SP2_MD5 = "afc70fe91069a85c416eba18b48ee30e"
OUT = HERE / "variants"

# sp2's book, in LAW_BOOK's and PATTERNS' order (the pose order: ties among candidates go by it)
LAWS = ("lt", "eq", "min", "max", "red", "div", "add", "sub", "mul", "acc.mul", "acc.add")
PATTERNS = ("pair", "x+n", "x+dn", "x+on", "Bx+d", "x,d", "x,o", "o,x,d")
SLUG = {"acc.mul": "accmul", "acc.add": "accadd", "x+n": "xn", "x+dn": "xdn", "x+on": "xon", "Bx+d": "bxd",
        "x,d": "xd", "x,o": "xo", "o,x,d": "oxd"}

# --------------------------------------------------------------------------------------------- the book
# SelfClimb poses only the laws and patterns it is given (its laws= and patterns= arguments, None: the whole book);
# sp2 passes neither. A book arm passes both, as tuples in the book's order.
AT_SP = "SP = dict(rule=SP_RULE, K=SP_K, NP=SP_NP,"
SP_END = "fuse=FUSE, round_retry=ROUND_RETRY)\n"


def book(laws=LAWS, patterns=PATTERNS, why=""):
    assert all(l in LAWS for l in laws) and all(p in PATTERNS for p in patterns)
    laws = tuple(l for l in LAWS if l in laws)
    patterns = tuple(p for p in PATTERNS if p in patterns)
    return [(AT_SP, f"SP_LAWS = {laws!r}  # ablation: the laws posed ({why})\n"
                    f"SP_PATTERNS = {patterns!r}  # ablation: the patterns posed\n" + AT_SP),
            (SP_END, "fuse=FUSE, round_retry=ROUND_RETRY,\n          laws=SP_LAWS, patterns=SP_PATTERNS)\n")]


# --------------------------------------------------------------------------------------------- the rule
RULE = 'SP_RULE = "lpca"       # the choosing rule:'
PICK = '    def pick(self, status, rows=None):\n        if self.rule == "all":\n'
CONCEPT_PICK = ('        if self.rule in ("lpc", "lpca"):              # concepts: a law on a pattern, hardest by its '
                'cheapest binding,\n'
                '            got = self.concepts(status, self.K, ties=self.rule == "lpca")     # practised once, in '
                'its cleanest\n')
VARIANTS_INIT = ('        self._variants = rule not in ("lpc", "lpca")    # pose practised concepts\' other bindings '
                 '(concept rules: no)\n')
VARIANTS_SLEEP = '            self._variants = self.rule not in ("lpc", "lpca") or self._want_variants\n'
RANKED = ('        ranked = sorted(by.values(), key=lambda v: (-min(c for _, c, _, _ in v), min(i for _, _, i, _ in '
          'v)))\n')
TIES = '        return [n for v in ranked[:k] for t, c, _, n in sorted(v) if (t, c) == min(v)[:2]]\n'
LOOK_ON = ('            if self.rule == "ct2a2" and (answers is not None or self.target is not None) and '
           'self.learnable(self._status):\n')
LOOK_OFF = '            if self.rule == "ct2a2" and not self.looks:\n'
LOOK_JOBS = '        jobs = [(n,) for n in learn]\n        jobs += ['
SCORES = 'scores=self._scores if self.rule in ("ct2a2", "lpu") else {}))'
SEEDED = "torch.Generator().manual_seed(torch.initial_seed() % 2 ** 31 * 1009 + self.level)"  # the run's seed (the evaluator's
#                                                                                       torch.manual_seed) and the level


def rule(name, why):
    return (RULE, f'SP_RULE = "{name}"  # ablation: {why}. The choosing rule (sp2: "lpca"):')


def concept_rule(name):
    """Make the rule `name` one of the concept rules: lpca's picking (every binding tied, by concepts()), and its
    posing (a practised concept's other bindings are not posed until no new concept is learnable)."""
    return [
        (CONCEPT_PICK, CONCEPT_PICK.replace('("lpc", "lpca")', f'("lpc", "lpca", "{name}")').replace(
            'ties=self.rule == "lpca")', f'ties=self.rule in ("lpca", "{name}"))')),
        (VARIANTS_INIT, VARIANTS_INIT.replace('("lpc", "lpca")', f'("lpc", "lpca", "{name}")')),
        (VARIANTS_SLEEP, VARIANTS_SLEEP.replace('("lpc", "lpca")', f'("lpc", "lpca", "{name}")')),
    ]


ARMS = {}
for _l in LAWS:
    ARMS[f"no_law_{SLUG.get(_l, _l)}"] = (f"the law {_l} out of the book (on every pattern)",
                                          book(laws=[l for l in LAWS if l != _l], why=f"sp2's without {_l}"))
for _p in PATTERNS:
    ARMS[f"no_pat_{SLUG.get(_p, _p)}"] = (f"the pattern {_p} out of the book (every law on it)",
                                          book(patterns=[p for p in PATTERNS if p != _p], why="all of sp2's"))

ARMS["rule_random_ok"] = ("the choosing rule: bar 3's random_ok (K learnable candidates at random, seeded by the run's "
                          "seed and the level) instead of lpca", [
    rule("random_ok", "bar 3's random_ok, K learnable candidates at random"),
    (PICK, '    def pick(self, status, rows=None):\n'
           '        if self.rule == "random_ok":                  # ablation: bar 3\'s random_ok, K learnable '
           'candidates at\n'
           '            learn = self.learnable(status)            # random (the run\'s seed and the level)\n'
           f'            g = {SEEDED}\n'
           '            return [learn[int(i)] for i in torch.randperm(len(learn), generator=g)[:self.K]]\n'
           '        if self.rule == "all":\n'),
])
ARMS["rule_rca"] = ("the choosing rule: lpca's concepts and bindings, the concepts taken in a random order (seeded by "
                    "the run's seed and the level) instead of hardest first", [
    rule("rca", "lpca with the concepts in a random order"),
    (RANKED, RANKED + '        if self.rule == "rca":                        # ablation: the concepts in a random '
                      'order\n'
                      f'            g = {SEEDED}\n'
                      '            ranked = [ranked[int(i)] for i in torch.randperm(len(ranked), generator=g)]\n'),
] + concept_rule("rca"))
ARMS["rule_lpcc"] = ("the choosing rule: lpca's concepts, hardest first, each practised in every binding tied for "
                     "cheapest instead of cleanest", [
    rule("lpcc", "lpca with each concept practised in its cheapest bindings, not its cleanest"),
    (TIES, '        if self.rule == "lpcc":                       # ablation: every binding tied for cheapest, however '
           'many\n'
           '            return [n for v in ranked[:k] for t, c, _, n in sorted(v, key=lambda e: (e[1], e[0], e[2]))\n'
           '                    if c == min(e[1] for e in v)]               # registers it touches\n' + TIES),
] + concept_rule("lpcc"))
ARMS["rule_ct1"] = ("the choosing rule: bar 3's ct1 (the round searched with each learnable candidate alone as a "
                    "routine; the K that shorten it most, ties in pool order) instead of lpca", [
    rule("ct1", "bar 3's ct1, the single lookahead"),
    (LOOK_ON, LOOK_ON.replace('self.rule == "ct2a2"', 'self.rule in ("ct2a2", "ct1")')),
    (LOOK_OFF, LOOK_OFF.replace('self.rule == "ct2a2"', 'self.rule in ("ct2a2", "ct1")')),
    (LOOK_JOBS, '        jobs = [(n,) for n in learn]\n'
                '        if self.rule == "ct1":                        # ablation: bar 3\'s ct1, every learnable '
                'alone\n'
                '            return jobs\n'
                '        jobs += ['),
    (PICK, '    def pick(self, status, rows=None):\n'
           '        if self.rule == "ct1":                        # ablation: bar 3\'s ct1, the K best single '
           'lookaheads\n'
           '            n_sym = len(PRIMS) + len(self.routines())  # (ties in pool order)\n'
           '            base = self.mdl(self._base_look, n_sym)\n'
           '            singles = {j[0]: base - self.mdl(r, n_sym + 1) for j, r in self.looks}\n'
           '            self._scores = dict(singles=singles)\n'
           '            return sorted(singles, key=lambda n: -singles[n])[:self.K]\n'
           '        if self.rule == "all":\n'),
    (SCORES, SCORES.replace('("ct2a2", "lpu")', '("ct2a2", "lpu", "ct1")')),
])
ARMS["rule_nov"] = ("the choosing rule: bar 3's novelty (on the target's inputs, the rows' N and x at full size as the "
                    "round starts: the rows where a candidate's A differs from every token's; the K most novel, ties "
                    "in pool order) instead of lpca", [
    rule("nov", "bar 3's novelty"),
    (PICK, '    def pick(self, status, rows=None):\n'
           '        if self.rule == "nov":                        # ablation: bar 3\'s novelty on the target\'s inputs '
           '(the\n'
           '            Nd, Xd = rows[0][:256], rows[1][:256]     # rows\' N and x at full size, as the round '
           'starts): the rows\n'
           '            P = self.P_round                          # where a candidate\'s A differs from every '
           'token\'s\n'
           '            st, ins = round_settings(self.cells, P), {"X": pad(Xd, P), "N": pad(Nd, P)}\n'
           '            out = lambda prog: run_prog(st, prog, self.lib, ins, gen_on(self.device, 5), self.device)[0]\n'
           '            seen = torch.stack([out((t,)) for t in list(PRIMS) + list(self.routines())])\n'
           '            learn = self.learnable(status)\n'
           '            score = {n: int((seen != out(status[n]["prog"])[None]).any(-1).all(0).sum()) for n in learn}\n'
           '            self._scores = dict(novelty=score)\n'
           '            return sorted(learn, key=lambda n: -score[n])[:self.K]\n'
           '        if self.rule == "all":\n'),
    (SCORES, SCORES.replace('("ct2a2", "lpu")', '("ct2a2", "lpu", "nov")')),
])


def build(arm: str, src: str) -> str:
    what, edits = ARMS[arm]
    out = f"# Ablation arm {arm} (integration/sp_ablation/SP_ABLATION.md): sp2 (md5 {SP2_MD5[:8]}) with one change: " \
          f"{what}.\n" + src
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{arm}: an edit's old text occurs {n} times, not once: {old[:80]!r}")
        out = out.replace(old, new)
    compile(out, f"{arm}.py", "exec")
    return out


SET_RULES = [("lp", "bar 3's hardest learnable first"), ("ct2a2", "bar 3's cheaper pairs signal"),
             ("all", "bar 3's reference: every learnable candidate"),
             ("lpu", "SELF_PRACTICE.md's unlock first, then hardest first"),
             ("lpc", "lpca with each concept in its one cleanest binding, not every tied one")]


def arms_json():
    """plans/arms.json: every arm's file (a variant, or sp2 itself with its --set) and what it changes."""
    sp2 = "integration/self_practice/submission_stage3_sp2.py"
    arms = {"control": {"file": sp2, "what": "sp2 unchanged"}}
    for r, what in SET_RULES:
        arms[f"rule_{r}"] = {"file": sp2, "set": {"SP_RULE": f'"{r}"'}, "what": f"the choosing rule: {what} instead "
                                                                                f"of lpca"}
    for arm, (what, _) in ARMS.items():
        arms[arm] = {"file": f"integration/sp_ablation/variants/{arm}.py", "what": what}
    (HERE / "plans" / "arms.json").write_text(json.dumps(arms, indent=1) + "\n")


def main(argv):
    src = SP2.read_text()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if md5 != SP2_MD5:
        raise SystemExit(f"{SP2} md5 {md5} != {SP2_MD5}: sp2 changed")
    if argv[:1] == ["--diff"]:
        out = build(argv[1], src)
        sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), out.splitlines(True), str(SP2.name),
                                                   f"variants/{argv[1]}.py", n=1))
        return
    OUT.mkdir(exist_ok=True)
    sums = {}
    for arm, (what, _) in ARMS.items():
        out = build(arm, src)
        path = OUT / f"{arm}.py"
        if path.exists() and path.read_text() != out and "--force" not in argv:
            raise SystemExit(f"{path} differs from this build (pass --force to replace it)")
        path.write_text(out)
        sums[arm] = {"md5": hashlib.md5(out.encode()).hexdigest(), "what": what}
        print(f"{arm:16s} {sums[arm]['md5']}  {what}")
    (OUT / "MD5SUMS.json").write_text(json.dumps({"sp2": SP2_MD5, "variants": sums}, indent=1) + "\n")
    arms_json()


if __name__ == "__main__":
    main(sys.argv[1:])
