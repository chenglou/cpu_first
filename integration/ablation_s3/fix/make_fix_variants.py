"""The fix of §8 (make_fix.py) carried into the eight certified variants that still have the old checks (ABLATION_S3.md
§8): full-size probes for the reduction, probes with N < x < 2N for the conditional subtraction. Writes each fixed copy
into a fix/ folder of its own study, from the certified file by make_fix.py's text edits (each asserted to occur exactly
once), and prints the md5s; reads the originals and never writes them, nor anything their rebuilds read.

- s3i's two variants (strict s3i, the bridge) take s3i's edits as they are. The bridge's addition cell is ADD, not
  ADC, but its table is s3i's adc table on Cells (above_probes' adc_d, adc_c), and the reduction's probes add with the
  learned helper "add", whatever instructions it uses.
- sp2's five (strict sp2, the learned loop, sp_breadth's g3, book_min, generic_ops' g3) take sp2's edits; strict sp2
  renamed search_s to search_work. generic_ops' g3 already accepts at four digits, which a loop of 10^4 subtractions
  (B loops do not grow with W, and GENERIC_OPS.md §2.1 finds such fakes at cost 5, long division's cost there) passes
  on every pool, so it takes both parts too.
- book_gen's generated book (sp_ablation) has its own Draw and law_target and a second reduction with a large
  quotient, red on x + x N (seed 74's round calls it): the same edits on its anchors, both reductions at full size.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/fix/make_fix_variants.py [--diff NAME]
"""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTEG = HERE.parent.parent
CPU_FIRST = INTEG.parent
sys.path.insert(0, str(HERE))
import make_fix as MF  # noqa: E402  (§8's edits)

S3I, SP2 = MF.S3I_EDITS, MF.SP2_EDITS

# strict sp2: make_strict.py renamed search_s (the search's seconds) to search_work (its machine work)
SP2_STRICT = [(o.replace("search_s", "search_work"), n.replace("search_s", "search_work")) if i == 7 else (o, n)
              for i, (o, n) in enumerate(SP2)]

# book_gen: the same changes on the generated book's Draw and law_target, and on both of its reductions whose quotient
# is another row's x or x itself
BG_DRAW = "        self.st, self.gen, self.device, self.pattern, self.w = st, gen, device, pattern, w\n        self.cache = {}\n"
BG_LAW = '    vary)."""\n    kinds = PATTERNS[d.pattern][0]\n'
BG_TERM = "            v = repeat_add(ops, st, lib, self.x, self.N, self.count(NFORM[name]), gen, dev)"
BOOK_GEN = list(SP2)
BOOK_GEN[0] = (SP2[0][0], SP2[0][1].replace(
    'FULL_CHECK = (("red", "x+on"),)  # fix: laws on patterns whose acceptance also runs at the round\'s size (W '
    'digits):\n',
    'FULL_CHECK = (("red", "x+on"), ("red", "x+xn"))  # fix: laws on patterns whose acceptance also runs at the '
    'round\'s size\n'
    '                                 # (W digits; both reductions whose quotient is another row\'s x or x itself):\n'))
BOOK_GEN[2] = (BG_DRAW, BG_DRAW + SP2[2][1][len(SP2[2][0]):])
BOOK_GEN[4] = (BG_LAW, BG_LAW.split("\n")[0] + "\n" + SP2[4][1][len(SP2[4][0]):] + BG_LAW.split("\n", 1)[1])
BOOK_GEN[6] = (BG_TERM,
               "            add_q = times_add if st.W == W else repeat_add      # fix: at the round's size by doubling\n"
               "            v = add_q(ops, st, lib, self.x, self.N, self.count(NFORM[name]), gen, dev)")
assert all(n.startswith(o) for o, n in (SP2[2], SP2[4])), "sp2's Draw and law_target edits only append"
assert BOOK_GEN[0] != SP2[0], "book_gen's FULL_CHECK"

S3I_WHAT = ("the reduction's acceptance also checked at the round's size (8 digits), its probes built by doubling, and "
            "the conditional subtraction's also with N < x < 2N")
SP2_WHAT = ("the reduction law's (red on x+on) acceptance also checked at the round's size (8 digits), its probes built "
            "by doubling, and the conditional subtraction's (red on pair) also with N < a < 2N")
FIXES = {   # name: (original, its md5, the copy's folder, edits, the header's description)
    "submission_stage3_s3i_strict_fix": (INTEG / "strict" / "submission_stage3_s3i_strict.py",
                                         "11baebb4265de98a395033ae90a3cebe", INTEG / "strict" / "fix", S3I, S3I_WHAT),
    "submission_stage3_sp2_strict_fix": (INTEG / "strict" / "submission_stage3_sp2_strict.py",
                                         "1a1f15fb231d6da5fd53d530bc9bdcb6", INTEG / "strict" / "fix", SP2_STRICT,
                                         SP2_WHAT),
    "submission_stage3_ll_fix": (INTEG / "learned_loop" / "submission_stage3_ll.py", "419c15e70ccf6483ff68b04b08f52bdc",
                                 INTEG / "learned_loop" / "fix", SP2, SP2_WHAT),
    "submission_stage3_bridge_fix": (CPU_FIRST / "bar3" / "bridge" / "submission_stage3_bridge.py",
                                     "85ff74c5faf902c1ede7a9f07d059026", CPU_FIRST / "bar3" / "bridge" / "fix", S3I,
                                     S3I_WHAT),
    "submission_spb_g3_fix": (INTEG / "sp_breadth" / "submission_spb_g3.py", "30efec64d6e4f0a7417938e9370a363e",
                              INTEG / "sp_breadth" / "fix", SP2, SP2_WHAT),
    "book_min_fix": (INTEG / "sp_ablation" / "variants" / "book_min.py", "e4732f09e8e5b46a8cfa5a1236771bd2",
                     INTEG / "sp_ablation" / "fix", SP2, SP2_WHAT),
    "book_gen_fix": (INTEG / "sp_ablation" / "variants" / "book_gen.py", "8bbac4a1c631a08c8b72179de37fcb26",
                     INTEG / "sp_ablation" / "fix", BOOK_GEN,
                     "the reductions' (red on x+on and on x+xn) acceptance also checked at the round's size (8 digits), "
                     "their probes built by doubling, and the conditional subtraction's (red on pair) also with "
                     "N < a < 2N"),
    "submission_stage3_g3_fix": (INTEG / "generic_ops" / "submission_stage3_g3.py", "e6bd2a1f9bfd9fd2119b0d44483c7100",
                                 INTEG / "generic_ops" / "fix", SP2,
                                 "the reduction law's (red on x+on) acceptance also checked at the round's size (8 "
                                 "digits; its own sizes reach 4), its probes built by doubling, and the conditional "
                                 "subtraction's (red on pair) also with N < a < 2N"),
}
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size


def build(name: str) -> tuple[str, str]:
    path, md5, _, edits, what = FIXES[name]
    src = path.read_text()
    got = hashlib.md5(src.encode()).hexdigest()
    if got != md5:
        raise SystemExit(f"{path} md5 {got} != {md5}: the original changed")
    out = f"# {name} (integration/ABLATION_S3.md §8): {path.name} (md5 {md5[:8]}) with {what}.\n" + src
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{name}: an edit's old text occurs {n} times, not once: {old[:80]!r}")
        out = out.replace(old, new)
    compile(out, f"{name}.py", "exec")
    if len(out.encode()) > LIMIT:
        raise SystemExit(f"{name}: {len(out.encode()):,} bytes, over the evaluator's 256 KiB")
    return src, out


def main(argv):
    if argv[:1] == ["--diff"]:
        src, out = build(argv[1])
        sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                                   FIXES[argv[1]][0].name, f"fix/{argv[1]}.py", n=1))
        return
    for name, (_, _, folder, _, _) in FIXES.items():
        _, out = build(name)
        path = folder / f"{name}.py"
        if path.exists() and path.read_text() != out and "--force" not in argv:
            raise SystemExit(f"{path} differs from this build (pass --force to replace it)")
        folder.mkdir(exist_ok=True)
        path.write_text(out)
        print(f"{name:34s} {hashlib.md5(out.encode()).hexdigest()}  {len(out.encode()):,} bytes  "
              f"{path.relative_to(CPU_FIRST)}")


if __name__ == "__main__":
    main(sys.argv[1:])
