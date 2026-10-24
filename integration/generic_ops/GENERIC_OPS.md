# Generic instructions in place of SEL.S and BRG.X

*Written 2026-10-24 for a judge of the contest and for the project's own bar 2 ([OVERVIEW.md](../../OVERVIEW.md) §7,
open question 1). Code, records and plans are in this folder. Nothing outside it changed, and no certified file was
touched (the builder asserts their md5s).*

<!-- PLAIN -->
**The question.** Stage 3's digit machine has two instructions that are long division's step written as instructions:
SEL.S ("keep or discard on the borrow": A := S where the last subtraction borrowed) and BRG.X ("bring the next digit
down": A's digits up one place, X's top digit into A). The s3i ablation found both needed. Can they be replaced by
generic instructions, so that the learner has to build the step itself?

**What replaced them.** SEL.S by an if block, `C[...]`: its body's effects stay only where the borrow was set (a
conditional skip; no instruction selects from a save register any more). BRG.X by shifts that pass the digit they
move through the multiply's carry digit k (SHL r: r := r·10 + k, and k := r's old top digit), so that "SHL.X then
SHL.A" moves X's top digit into A. No single instruction does either old job. A third replacement, no conditional at
all (the borrow moved into the multiplier digit), reaches the round only with deep probes seven times larger, after
3.6 hours of the climb alone: far past the hour the contest allows.

**Does it still find the round? Yes, and it is certified.** The learner built the step itself (§4): the conditional
subtraction as `T.sub.pair.YX C[LDA.Y]` (subtract; if it borrowed, reload the input), the chain as
`B[T.lt C[T.add]]` (subtract; if it borrowed, add back), long division as `LDA.N P[SHL.Y SHL.A T.chain]` (the two
shifts as its "bring down"), and, new to stage 3, a shift-and-add multiply over the accumulator. In the contest's own
evaluation (official manifest, one H100, seed 74) the file `submission_stage3_g3.py` (md5 `e6bd2a1f`) found its round
after level 4, by 2,507 s of training, froze by 2,572 s, and is certified to T = 64 on both ladders, with every
split exact (test 9,999 of 9,999; unseen T and unseen moduli 10,002 of 10,002 each). Its 8 seeds on B200 (74 and 1 to
7) certified the same way, each with its round after level 4. `tools/verify_records.py` checks all nine records
against the file (§0).

**What it took.** The instruction set alone is not enough. Replacing SEL.S alone works with no other change (the round
after level 4 in the tiny core, 15% slower). Replacing BRG.X makes long division one step longer (cost 5, not 4), and
three things then stop the climb, each found in a run and fixed by a rule that holds for every law:
1. cheaper fakes of the reduction (100 or 1,000 conditional subtractions) pass the practice probes, and sp2's gate
   then scores the reduction as "reached" and never searches deeper: fixed by counting closeness without the fakes,
   giving two concepts a deep probe a level, rotating the deep probes to concepts whose closeness grew, and searching
   the round by calls first;
2. a fake that passes acceptance at three digits beats the real program: fixed by acceptance at four digits;
3. without SEL.S, saving into S costs a step more, so the learner's chains borrow X or Y as scratch, and long division
   needs that register intact: fixed by preferring, one cost further, routines that leave every register but their
   own operands alone.
The law book did not change.

**At what cost?** About 2.4 times sp2's training time to the round on the official run (by 2,507 s against 1,029 s),
and 2.2 times its whole training (2,770 s against 1,277 s), inside the 3,600 s budget with 830 s to spare.
On the 8 B200 seeds the round came between 2,042 and 2,505 s (sp2 on the same seeds: 650 to 1,775 s), and training
took 2,290 to 2,790 s; in the tiny core, 1.3 to 2.4 times sp2's. The rules were found by watching whether the round
came, which is design with the target in view, and the first of them also costs sp2's own machine a level. About $255
of compute.
<!-- /PLAIN -->

## 0. Status, records and how they were run

**The candidate.** `integration/generic_ops/submission_stage3_g3.py`, md5 `e6bd2a1f9bfd9fd2119b0d44483c7100`, 244,664
bytes, instruction set `if1,k`. §8 has the command that rebuilds it byte for byte.

**Certified.** Its official H100 run and its 8 B200 seeds ran on 2026-10-24 from 14:42 PDT (runs `g3-official-2`
and `g3-seeds-2`). All nine certified to T = 64 on both ladders (seen and unseen moduli), with every split exact
(test 9,999 of 9,999; unseen T and unseen moduli 10,002 of 10,002 each). Each found its round by calls after level 4,
at cost 2, and froze about 200 steps later:

| run | GPU, seed | the round: step; training clock | frozen: step; clock | training | evaluation | wall | sp2 on the same GPU and seed: round; training |
|---|---|---|---|---|---|---|---|
| official | H100, 74 | 698; 2,507 s | 897; 2,572 s | 2,770 s | 386 s | 3,163 s | 1,029 s; 1,277 s |
| seed | B200, 74 | 698; 2,505 s | 897; 2,570 s | 2,790 s | 381 s | 3,178 s | 702 s; 855 s |
| seed | B200, 1 | 640; 2,207 s | 839; 2,247 s | 2,444 s | 302 s | 2,752 s | 650 s; 786 s |
| seed | B200, 2 | 689; 2,362 s | 888; 2,416 s | 2,593 s | 306 s | 2,905 s | 744 s; 886 s |
| seed | B200, 3 | 610; 2,104 s | 809; 2,135 s | 2,338 s | 298 s | 2,644 s | 1,775 s; 2,010 s |
| seed | B200, 4 | 703; 2,234 s | 902; 2,261 s | 2,426 s | 291 s | 2,723 s | 937 s; 1,209 s |
| seed | B200, 5 | 643; 2,157 s | 842; 2,197 s | 2,388 s | 332 s | 2,726 s | 651 s; 784 s |
| seed | B200, 6 | 607; 2,214 s | 806; 2,243 s | 2,419 s | 299 s | 2,724 s | 818 s; 1,073 s |
| seed | B200, 7 | 688; 2,042 s | 887; 2,095 s | 2,290 s | 307 s | 2,606 s | 931 s; 1,187 s |

- The clock is the training clock at the first progress line (one every 100 steps) at or after the step, so the round
  and the freeze had happened by then. sp2's figures are from its records in `../self_practice/records/runner/`.
- Every round is four programs of one form: x into A (and Y), the multiply, the product into X or Y, the reduction by
  N. On seed 74, two of the four use the shift-and-add multiply (`LDA.X T.mul.x,o.AX STA.Y T.red.x+on.YN`).
- On B200, g3's median time to the round is 2.8 times sp2's (2,211 s against 781 s; seed by seed, 1.2 to 3.6
  times), and its median training 2.5 times (2,423 s against 980 s). The slowest of the nine trained 2,790 s of the
  3,600 s budget (sp2's slowest: 2,010 s).
- Peak host memory 13.0 to 14.1 GiB.

**Verified.** `tools/verify_records.py` checks all nine records. Each embeds the file's source, md5
`e6bd2a1f9bfd9fd2119b0d44483c7100`, with the official manifest (seeds 1 to 7: official except the seed), return
code 0 and a RESULT_JSON line equal to its result:

```bash
cd cpu_first && python3 tools/verify_records.py integration/generic_ops/records/runner/submission_stage3_g3_official.json=integration/generic_ops/submission_stage3_g3.py integration/generic_ops/records/runner/submission_stage3_g3_b200_seed*.json
```

**The first attempt, stopped.** At 07:27 PDT the same day, Modal stopped every container in the workspace at its
spend limit, and refused launches until the limit was raised:
- the first official run (run `submission-stage3-g3-official`) was stopped during evaluation and wrote no record. By
  its log (the last copy is `records/runner/partial/submission-stage3-g3-official.peek.txt`), it had found the round
  at step 698 (step 700 at 2,373 s of training), resolved T's direction at step 718 and frozen at step 897, the same
  steps as the rerun on the same seed. It had certified seen T = 1 to 64 and unseen T = 1, 2, 4;
- its 8 B200 seeds (run `g3-seeds`) were stopped six minutes in (steps 500 to 600), with no records;
- the official run of `submission_stage3_g_brg3.py` (run `submission-stage3-g-brg3-official`) was stopped at level 5,
  about 2,950 s in, with no record (§5). It is not needed for the result and was not relaunched.

The three stopped apps (ap-4CbUBB9ywEgDf0ckoZNLna, ap-s0kQgp7W8LLMBLN2hiaa43, ap-JhxvAGCpX7YUGC72Ggg5iC) were left as
they were, and none had run again by the time the reruns finished.

**Complete records** (runs that ended, one way or another):
- tiny core: `records/<run>.json` and `.log` for the 21 runs of §3 that ran to their end (`"final": true` in the
  record), plus `g_brg3`, which the stop cut off at level 6 (2,142 s; its last snapshot on the Volume). Also:
  - `records/test_isa_b200.log`;
  - the gate diagnoses `records/diag_gate_if_k_L4.log` and `records/diag_gate_brg3_L4.log`, made from the record
    snapshots in `diag/`;
- evaluator, in `records/runner/`:
  - `submission_stage3_g3_official.json` and `submission_stage3_g3_b200_seed{74,1,2,3,4,5,6,7}.json`: g3, certified
    (above);
  - `submission_stage3_g_brg_official.json` and `submission_stage3_g_official.json`, both killed by the runner's
    memory cap;
  - `submission_stage3_g2_official.json` and `submission_stage3_g_brg2_official.json`, both of which ran their
    3,600 s and are not certified;
  - `submission_stage3_g_b200_seed74.json`, an earlier build (md5 `dad32aa5`), not certified.

**How they were run** (about $43 at list price for the nine: $3.56 for the H100 run, $39.28 for the eight B200
seeds). The spend check, then the two launches. They took `--force` because the stopped runs held the record names:

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 3750
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/generic_ops/submission_stage3_g3.py --memory-gb 48 --out integration/generic_ops/records/runner --run g3-official-2 --force
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/generic_ops/submission_stage3_g3.py --seeds 8 --hardware b200 --memory-gb 48 --out integration/generic_ops/records/runner --run g3-seeds-2 --force
```

Then the fetches, each of which waits until its records are in:

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python tools/modal_runner.py --mode fetch --run g3-official-2 --wait
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python tools/modal_runner.py --mode fetch --run g3-seeds-2 --wait
```

A rerun needs new run tags. Give it another `--out` as well, so that the records above keep their place (a fetch
replaces a record from another run only with `--force`).

## 1. The two instructions, and what replaces them

Stage 3's digit machine ([`lib/libcore.py`](../lib/libcore.py)) has five registers of P digits (A, X, Y, N, S), one
pointer shared by all of them, a borrow/carry bit c, a multiplier digit h and a multiply carry digit k. Its 32
instructions read the learned cells at the pointer (SBB, ADC, MUL), move whole registers (LDA, STA, CLA, the shifts),
clear the flags, load h, or move the pointer, and loops run a body P, B (the base) or W times. Two of the 32 are long
division's step written as instructions:

- **SEL.S**: A := S wherever the borrow is set. With "save A in S, subtract N" in front it is the restoring step: keep
  the difference, or take the saved value back when the subtraction borrowed. It is the machine's only conditional,
  and S is the register it exists for.
- **BRG.X**: A's digits move up one place, X's top digit comes into A's lowest place, and X's digits move up one place.
  It is long division's "bring the next digit down".

The s3i ablation ([ABLATION_S3.md](../ABLATION_S3.md)) found both needed: without SEL.S no conditional subtraction is
found (the climb ends after level 1), and without BRG.X no step and no reduction.

This study takes both out and puts generic pieces in, so that the learner has to build the two steps itself. Three
replacements were built, from more to less generic:

| replaces | instruction set | what it is | a CPU's counterpart |
|---|---|---|---|
| SEL.S | **`ldhc`** | LDH.C: h := the borrow. No conditional at all: a multiply-accumulate by h adds a register or adds nothing | moving the carry into a register (x86 `SETC`) |
| SEL.S | **`if`**, **`if1`** | C[...]: a block whose effects stay only where the borrow was set at its start. The pointer moves the same everywhere; registers and flags keep their old values where the borrow was clear. In `if1` the searches only make blocks of one item of cost 1 (an instruction, a routine call or a merge): "skip the next instruction unless the borrow is set" | predicated execution (ARM's `IT`), a skip on carry (PDP-8 `SNL`) |
| BRG.X | **`k`** | the shifts pass through the multiply's carry digit k: SHL r makes r := r·B + k and k := r's old top digit; SHR r the reverse. A shift is the multiply (or divide) by the base, with the carry the multiply already has. "SHL.X then SHL.A" moves X's top digit into A | shift and rotate through carry (x86 `RCL`/`RCR`) |

No single instruction of these does either old job. C[LDA.S] does what SEL.S did, but it is a generic `if` around a
plain load, and the search has to choose that body. SHL.A alone brings in whatever k holds, and only after SHL.X has
put X's top digit there does the pair act as BRG.X. Naming: `ISA = "COND,SHIFT[,brg]"`, so `if1,k` is if blocks of
one item with shifts through k, and `sel,zero,brg` is libcore.py's own set (SEL.S, zero-fill shifts, BRG.X).

### 1.1 The machine, four times over

Every machine that runs programs needed the new pieces, and each was checked against the plain one (`run_items`):

- `step` / `run_items` (the reference: acceptance, the practice problems' targets, side effects);
- the lockstep machine (`_ustep`, `apply_items`: the searches, on CUDA graphs on a GPU). A C block becomes two
  opcodes; each probe counts the blocks it is "off" in (a block starts off where the borrow is clear, or inside a block
  that is off), and every write is masked by that count;
- the round machine (`RoundMachine`: phase D's checks on CUDA graphs): a block saves the state and its borrow at the
  start and restores the state where the borrow was clear;
- the one-hot machine (`SteMachine`: the prediction path, inside autograd): every value a block changed is mixed with
  its old value by the borrow's one-hot, as SEL.S's soft select was.

[`test_isa.py`](test_isa.py) runs all four on every instruction set: every primitive, the loop vocabulary to body cost
2, and a library with C blocks and shifts through k nested in loops and in each other, at the climb's settings and at
16 places; 41 programs on the round machine and 25 on the one-hot machine at full size; and each set's hand-written
round on 400 of H1's T = 1 rows. Everything agrees, on a CPU and on a B200 with the CUDA graphs
(`records/test_isa_b200.log`). The hand-written pieces cost:

| instruction set | link (conditional subtraction) | reduction (long division over the chain) |
|---|---|---|
| `sel,zero,brg` (libcore.py) | `STA.S T.lt SEL.S`, cost 3 | `STA.Y P[BRG.X T.chain]`, cost 4 |
| `if`/`if1,k` | `STA.S T.lt C[LDA.S]`, cost 4 | `STA.Y P[SHL.X SHL.A T.chain]`, cost 5 |
| `ldhc,k` | `T.lt LDH.C CLK P[MUL.Y INC]`, cost 6 | as `if,k`, cost 5 |

## 2. Why the generic shifts at first found no reduction, and the fix

With BRG.X gone, long division costs one step more: `STA.Y P[SHL.X SHL.A T.chain]` is cost 5, where
`STA.Y P[BRG.X T.chain]` was cost 4. At level 4 the ordinary probe (1M states) stores the programs up to cost 3 and
streams cost 4, so it cannot reach cost 5. That is what the deep probes are for (5.8M states, a smaller vocabulary).
But the deep probes never came to the reduction: in both runs with the shifts through k and sp2's own rules (`g_sel_k`,
`g_if_k`), the reduction's closeness read 0.0 at every level from 3 on, so the gate sent the deep probes elsewhere (the
quotient, `eq`). There was still no real reduction after 8 levels (`g_if_k`, stopped by its time limit) or after 10
(`g_sel_k`, which found only a fake one at level 9) (§3).

[`diag_gate.py`](diag_gate.py) replays `g_if_k`'s level-4 probe of the reduction from its record
(`records/diag_gate_if_k_L4.log`). The probe does find goal states, but only fakes:

- at cost 3 and 4, `LDA.X B[T.red.x+dn.AY]` and `STA.Y LDA.X B[T.red.x+dn.AY]`: the chain (ten conditional
  subtractions) done ten times, so 100 subtractions. That is exact for every dividend with a quotient below 100,
  which is every probe at the two probe widths (one- and two-digit numbers), and wrong at the third width that
  acceptance adds (three digits). Acceptance rejects them;
- sp2's gate scores a law whose probe found only rejected programs as "reached, not out of reach", closeness 0
  ([SELF_PRACTICE.md](../SELF_PRACTICE.md) §5, item 2, a rule made to keep `eq` from taking every deep probe).

So the reduction looked solved-but-wrong, never far away, and was never searched deeper. Counted over the probe's held
states that are not goal states, its closeness was 0.54 to 0.77, the highest of any open law at that level.

**The fix** (`alias_fit`, a SelfClimb option; `SP_ALIAS_FIT` in the file): a law whose probe found only rejected
programs keeps its closeness, computed over its held states that are not goal states. It holds for every law alike
and names none. With it, the reduction gets its deep probe at level 4 and is found there (§3). It is a change to
sp2's choosing machinery, not to the law book.

It has a price: it changes which concepts the deep probes go to at every level. On the original instruction set it
delays the multiply by a level (`g_base_af`: the round after level 5 in 1,763 s, against after level 4 in 898 s
without it).

### 2.1 A second kind of fake, in the evaluator

The tiny core trains nothing: its rows are drawn once and its cells are exact. Inside the evaluator the climb poses its
problems on the batches' rows, so it can take another path, and on seed 74 it did. Both evaluator runs on that seed
with sp2's acceptance practised, at level 4, a fake reduction instead of long division:

- with BRG.X kept (`submission_stage3_g_brg.py`, official H100 run): `red/x+on@XN = B[T.red.x+dn.YX B[T.red.x+dn.XN]]`,
  cost 4, against `STA.Y P[BRG.X T.chain]`, also cost 4;
- with the shifts through k (an earlier build of `submission_stage3_g.py`, md5 `dad32aa5`, a B200 run whose record
  keeps its source):
  `B[T.red.x+n.YX B[T.red.x+dn.XN]]`, cost 4, against long division's 5.

Both are about a thousand conditional subtractions: exact for every quotient below 1,000, which covers every
acceptance row (dividends of up to three digits' quotient), and wrong at H1's size. With the if block the learner's
links keep their input in X or Y (`T.sub.pair.YX C[LDA.Y]` restores from Y), so its chains end up in bindings like X
against N and Y against X, and two such chains compose into the fake. sp2's links, `STA.S T.lt SEL.S`, work in A, and
the same compositions do not exist there. The round's search then finds no round with the fake, and the climb goes on
to level 5, whose vocabulary is far larger.

The fix is acceptance at one more size (`SP_VERIFY` to four digits): a fake with 1,000 subtractions fails it, and the
cheapest fake that passes needs ten times as many, one more B[...] around it. With BRG.X kept, that is cost 5 against
long division's 4, so the real program is the cheaper; with the shifts through k it is cost 5 against 5, a tie that
can go either way. The four-digit targets are built as every target is, by the learner's own additions (up to 9,999
of them for x + o N), which costs little next to the searches.

The same change closes the weakness the s3i ablation found in this check (`integration/ablation_s3/reduce_check.py`).
Acceptance tests 60 rows at each size, and a row's quotient is another row's x digits at N's top places. The
ablation's fake, `B[P[...]]`, does 10 × 2W × 10 subtractions: 600 at three digits. It passes three-digit acceptance
whenever none of the 60 rows has a quotient above 600, which happens on 3.0% of draws from H1's rows. At four digits
it does 800 subtractions, and the chance that all 60 rows have a quotient of 800 or less is 2.5e-49; for the
thousand-subtraction fakes above it is 1.6e-43. A fake would need about 8,000 subtractions to pass four digits two
times in three. [`accept_odds.py`](accept_odds.py) computes these figures.

A third finding of the same run: a level-5 probe unit spent 1,599 s building its composition vocabulary in plain Python
(the bodies of every loop over 78 tokens) without looking at the deadline, and the run trained 4,408 s against the
3,600 s budget. libcore_g's enumeration now checks the deadline every 65,536 bodies (`enumerate_items`); nothing else
changes when there is time.

### 2.2 Why the path matters: scratch registers

Whether a reduction exists at cost 5 (shifts through k) or 4 (BRG.X) depends on the chains the learner practised at
level 3, and those depend on its links at level 1:

- in `g_if1_gain_root` the learner practised one link, `T.sub.pair.YX C[LDA.Y]`; at level 3 the cheapest chain on A
  against Y was then the add-back `B[T.lt.pair.AY C[T.add.pair.AY]]` (cost 4), which writes no register but A, so long
  division can loop over it with its dividend in X or Y;
- in the four-width run (`g_if1_gain_root_v4`) and in the evaluator runs it practised more links (`LDA.X T.lt.pair.AY
  C[LDA.X]` in several bindings). At level 3 every binding then had a cost-3 chain that keeps its running value in
  an operand register, `B[STA.X T.red.pair.XY]` (X as scratch). The search stops at the first cost that has an
  accepted program, so the clean cost-4 chain is never found, and these chains overwrite the register that holds the
  dividend's remaining digits. Long division over them does not work, in any binding but the one whose scratch is
  free (dividend in Y, chain scratching X, modulus in N), and with BRG.X (which reads only X) in none. (At level 5
  the four-width run found another kind of reduction over such a chain; see §3.)

sp2 never meets this, because its link, `STA.S T.lt SEL.S`, scratches in S, and S is never one of a law's operands
(A, X, Y, N). So SEL.S gave the learner more than keep-or-discard: a scratch register outside the operands, which made
its chains compose with anything. With a generic conditional that is no longer free: the S-based link
`STA.S T.lt C[LDA.S]` costs 4, one more than the links that restore from their input register.

**The fix** (`pure_first`, `SP_PURE_FIRST`): when every program a practice problem accepts at its first cost changes a
register outside A and the problem's own operands (scratch), the search goes one cost further for one that does not,
and keeps the scratching ones only if none turns up. It is bar 3's rule for routines ("leave the inputs as they were,
change the fewest registers"), which sp2 applies among the shortest programs, applied across one more cost. It names
no law. On the path that failed (the four-width run, same six links at level 1), the chains at level 3 become the
clean add-back ones, `B[T.lt.pair.AX C[T.add.pair.AX]]` and its AY and AN twins (cost 4), and long division is found
at level 4 in four bindings (`g_pure_v4`). Two variants were run: "strict" (only A may change) and "scratch" (A and
the problem's own operands may change, so a multiply or a reduction may consume its inputs).

## 3. The tiny core: which sets reach the round, and with which search rules

The tiny core ([`core_g.py`](core_g.py), self_practice/core.py on libcore_g) runs the climb alone, offline, on the
data's rows with exact cells, one B200 each, with sp2's settings unless the table says otherwise (`lpca`, 4 concepts
a level, 8 probes a setting, probes of 1M, deep probes of 5.8M, the round's search to cost 5 at 4M states, then by
calls). `g_base` is libcore.py's instruction set and replays sp2's tiny-core run (`core_lpca_v13`) unit for unit, in
the same order with the same programs. Seconds are a B200's; "the round" is the level after whose sleep the round's
search found it. Records: `records/<run>.json` and `.log`; [`summarize_g.py`](summarize_g.py) prints them.

<!-- CORE -->
| run | instruction set | search rules changed from sp2 | the round | levels, seconds |
|---|---|---|---|---|
| `g_base` | sel,zero,brg (libcore.py) | none | after level 4 | 5 levels, 898 s |
| `g_if_brg` | if,zero,brg: SEL.S replaced | none | after level 4 | 5, 1,031 s |
| `g_sel_k` | sel,k: BRG.X replaced | none | none: no reduction for 8 levels, then a fake one at level 9 | 10 (the cap), 4,817 s |
| `g_if_k` | if,k: both replaced | none | none: no reduction | 8, 5,432 s (time limit) |
| `g_ldhc_k` | ldhc,k: no conditional | none | none: no conditional subtraction (it costs 6) | 10, 1,472 s |
| `g_ldhc_k_deep40` | ldhc,k | deep probes of 40M, one root a group | none: the link at level 1 (cost 6), the chain at level 4, no reduction | 7, 14,403 s (time limit) |
| `g_ldhc_k_af_deep40` | ldhc,k | as `g_ldhc_k_deep40`, + the alias fix | **after level 8**: long division at level 5, the multiply at level 6 | 9, 13,112 s |
| `g_base_af` | sel,zero,brg | the alias fix (§2) | after level 5 | 6, 1,763 s |
| `g_sel_k_af` | sel,k | the alias fix | after level 5 | 6, 3,066 s |
| `g_if_k_af` | if,k | the alias fix | none: the reduction at level 4, then every deep probe on `eq` | 5, 5,412 s (time limit) |
| `g_if_k_af_cf` | if,k | + the round by calls first | none: as `g_if_k_af` | 5, 5,420 s (time limit) |
| `g_if1_af_cf` | if1,k | the alias fix, by calls first | none: as `g_if_k_af` | 5, 5,518 s (time limit) |
| `g_if1_af_cf_r2` | if1,k | + the flat round search at 2M | none: as `g_if_k_af` | 5, 5,624 s (time limit) |
| `g_if1_af_cf_dk16` | if1,k | the alias fix, by calls first, two concepts' deep probes a level | none: the second slot went to `eq` | 5, 5,484 s (time limit) |
| `g_if1_af_cf_dk16_rot` | if1,k | + rotation "once" | after level 5 | 6, 3,944 s |
| `g_if1_gain` | if1,k | the alias fix, by calls first, two concepts, rotation "gain", flat search at 2M | **after level 4** | 5, 1,390 s |
| `g_if1_gain_root` | if1,k | as `g_if1_gain`, each deep probe's vocabulary from its own root | **after level 4** | 5, **1,205 s** |
| `g_if1_gain_root_v4` | if1,k | as `g_if1_gain_root`, acceptance to four digits | after level 5: scratch chains and no reduction at level 4 (§2.2), another kind of reduction at level 5 | 6, 3,219 s |
| `g_if1_brg` | if1,zero,brg: SEL.S replaced | none | after level 4 | 5, 912 s |
| `g_brg3` | if1,zero,brg | sp2's rules, + acceptance to four digits and side-effect-free routines ("scratch") | none by level 6: no reduction; cut off by the workspace stop | 6, 2,142 s (cut off) |
| `g_pure_v4` | if1,k | as `g_if1_gain_root_v4`, + side-effect-free routines (strict: only A may change) | **after level 4** | 5, 1,782 s |
| `g_scratch_v4` | if1,k | as `g_pure_v4`, "scratch" (A and the problem's own operands may change) | **after level 4** | 5, 2,183 s |
<!-- /CORE -->

What the table says:
- **SEL.S can go with no other change.** With the if block in its place (`g_if_brg`), the climb takes the same course
  as sp2's and finds the same round after level 4, 15% slower (1,031 s against 898 s). The learner builds the
  conditional subtraction itself, in two ways: `T.sub.pair.YX C[LDA.Y]` (Y - X, and where that borrowed, Y again: the
  input register is the saved copy, so S is not needed) and, for the chain, `T.lt.pair.AY C[T.add.pair.AY]` (subtract
  Y, and where that borrowed, add it back: the non-restoring step).
- **No conditional at all reaches the round, but far past the budget.** With LDH.C the conditional subtraction costs 6
  (`T.lt.pair.AX LDH.C CLK P[MUL.X INC]`: subtract, move the borrow into the multiplier, add X back times the
  borrow), and sp2's probes reach cost 5. Deep probes 7 times larger find it at level 1, after 1,230 s instead of
  190 s.
  - With sp2's rules the climb then stalls at the reduction (`g_ldhc_k_deep40`), like every run with the shifts
    through k.
  - With the alias fix as well (`g_ldhc_k_af_deep40`) it finds:
    - the chain at level 4 (`B[T.red.pair.AX]`: ten of those multiply-by-the-borrow steps);
    - long division at level 5 (`LDA.Y P[SHL.X SHL.A T.red.x+dn.AY]`);
    - the multiply at level 6;
    - the round after level 8.
  - That took 13,112 s, 3.6 hours of the climb alone, against the evaluator's one hour for everything.
- **BRG.X can go, but not with sp2's search rules.** Long division costs 5 with the shifts through k, one step more
  than with BRG.X. The ordinary probe holds programs up to cost 3 at level 4 and streams cost 4, so the reduction
  needs a deep probe, and sp2's gate never sends one there (§2). Four generic changes to the search rules, each tried
  because the one before was not enough, get it after level 4 (`g_if1_gain_root`):
  1. the alias fix (§2);
  2. the round searched by calls first (the flat search finds nothing here and cost up to 217 s a level), and the flat
     search at 2M states instead of 4M;
  3. two concepts' deep probes a level instead of one (`SP_DEEP_K` 16: whole concepts until 16 roots, so two of 12
     bindings), because the multiply and the reduction both need theirs at level 4;
  4. rotation "gain": a concept that already had a deep probe goes after the others unless its closeness has grown
     since. Without it the second slot goes to `eq` (closeness 0.8 at every level, never solved); the stricter
     "once" (a concept that had one goes after those that never had one) sent level 3's slot to the reduction before
     its chain existed, and then demoted it at level 4.

  The fifth, each deep probe's vocabulary from its own root's closest attempt (with the routines that calls) instead
  of every binding's, only saves time: 1,205 s against 1,390 s. sp2 took the union to rescue seed 7's multiply in
  the A bindings; with the shifts through k the multiply costs 5 there too (below).
- **Four-digit acceptance alone changes the path.** `g_if1_gain_root_v4` practised scratch chains at level 3 (§2.2),
  so no long division at level 4.
  - At level 5 it found another kind of reduction, `T.mul.x,o.XY W[SHR.X T.red.x+dn.YX]`, and the round after it
    (3,219 s).
  - That program reads as long division that moves the modulus instead of the dividend. The multiply routine runs for
    its side effect of shifting N up W places; then, W times, N shifts down one place and the scratch chain runs on Y.
  - It was not checked beyond the climb's acceptance and the round's search.
  - Side-effect-free routines first bring the round back to level 4 (`g_pure_v4`, `g_scratch_v4`).

## 4. What the climb built

With SEL.S and BRG.X gone, every piece of the round is a program the learner found and kept, at the level shown, in
`g_if1_gain_root` (the full-generic tiny core):

| piece | level | program | in words |
|---|---|---|---|
| the link (conditional subtraction) | 1 | `T.sub.pair.YX C[LDA.Y]` | A := Y - X; where that borrowed, A := Y again (the input register is the saved copy) |
| the chain | 3 | `B[T.lt.pair.AY C[T.add.pair.AY]]` | ten times: subtract Y; where that borrowed, add Y back (restoring division's step, as an add-back) |
| the multiply's row | 3 | `LDH.Y CLK P[MUL.X INC]` | A + X times Y's digit (as in s3i and sp2) |
| the multiply | 4 | `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]`, and `CLK P[SHL.A SHL.Y T.acc.mul.o,x,d.AXY]` | schoolbook rows (as in sp2); and shift-and-add from the top digit: x sits in A's low places, each pass shifts A up through k so that x's top digit drops into Y, then adds X times that digit, the hardware multiply loop |
| the reduction | 4 | `LDA.N P[SHL.Y SHL.A T.red.x+dn.AN]` | the modulus into A, then for each place: Y's top digit into k, k into A, the chain: long division, with SHL.Y SHL.A as its "bring the next digit down" |
| the round | after 4 | `LDA.X STA.Y T.mul.x,o.XY STA.Y T.red.x+on.YN` | x into Y, multiply, the product into Y, reduce it by N (found by calls: 2 calls) |

No instruction in it selects, restores or brings a digit in: the "keep or discard" is `C[...]` around an addition or a
load, and the "bring the next digit down" is two shifts. The shift-and-add multiply is new: no stage-3 run had it,
because a zero-fill shift loses the digit it moves out.

The certified official run (§0) built the same pieces, in its own bindings (its library at the round, in the record's
log):
- the link `LDA.Y T.lt.pair.AX C[LDA.Y]`;
- the chain `B[T.lt.pair.AN C[T.add.pair.AN]]`;
- the reduction `LDA.N P[SHL.Y SHL.A T.red.x+dn.AN]`;
- both multiplies, `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]` and `CLK P[SHL.A SHL.Y T.acc.mul.o,x,d.AXY]`.

The table's round, `LDA.X STA.Y T.mul.x,o.XY STA.Y T.red.x+on.YN`, is one of its round's four programs.

## 5. In the evaluator: the files and their runs

Every file is built by [`make_g.py`](make_g.py) from the template, libcore_g and spcore_g (§8), and checked locally
before its run (the static source policy, `designed_g.py` on four splits up to T = 64, and `smoke_g.py`). Runs go
through the shared runner, on one H100 with the untouched official manifest, seed 74 (records in `records/runner/`).

<!-- OFFICIAL -->
| file (md5) | instruction set, search rules | what happened | record |
|---|---|---|---|
| `submission_stage3_g_brg.py` (`2d9bb6ab`) | if1,zero,brg: SEL.S replaced; sp2's rules | the fake reduction `B[T.red.x+dn.YX B[T.red.x+dn.XN]]` practised at level 4, so no round; level 5's vocabulary took the host to 69.9 GiB and the runner killed it (cap 48 GiB) | `submission_stage3_g_brg_official.json` (killed) |
| `submission_stage3_g.py` (`038e70d1`) | if1,k, the five rule changes of §3 | the multiply at level 4, no reduction (scratch chains, §2.2); level 5 ran past the deadline (4,994 s) to 49.4 GiB, killed | `submission_stage3_g_official.json` (killed) |
| `submission_stage3_g2.py` (`1f16bf80`) | as `g`, + acceptance to four digits, + the deadline check | the multiply at level 4, no reduction; climbed to level 7, no round; trained exactly 3,600 s, 15.9 GB at most | `submission_stage3_g2_official.json`: not certified (test 7/9,999) |
| `submission_stage3_g_brg2.py` (`dc154a11`) | as `g_brg`, + four digits, + the deadline check | the fake now rejected, but no real reduction either (every chain on A scratched in X, which BRG.X reads); level 7, no round, 3,600 s | `submission_stage3_g_brg2_official.json`: not certified |
| **`submission_stage3_g3.py` (`e6bd2a1f`)** | **if1,k, the five rule changes, four digits, side-effect-free routines ("scratch")** | **the round found by calls after level 4 (step 698, by 2,507 s of training): `LDA.X STA.Y T.mul.x,o.XY STA.Y T.red.x+on.YN` and three others; T's direction at step 718; frozen at step 897 (by 2,572 s); training 2,770 s, evaluation 386 s. The first run (stopped by the workspace's spend limit at 07:27 PDT, mid-evaluation) had taken the same steps** | **`submission_stage3_g3_official.json`: certified, T = 64 on both ladders, every split exact** (the first run's log tail is in `records/runner/partial/`) |
| `submission_stage3_g_brg3.py` (`e18241d4`) | as `g_brg2`, + side-effect-free routines | clean chains in the run, but no reduction at level 4: the cost-4 long division is in the vocabulary and passes acceptance, yet the probe listed only fakes there, and with sp2's alias rule its closeness is 0, so no deep probe. Level 5's deep probes (sent by the quotient's closeness) found a reduction at cost 5 in three bindings, at about 2,950 s of the run, and the workspace cap killed it before the round's search | none (the log's tail is in `records/runner/partial/`) |

An earlier build of `submission_stage3_g.py` (`dad32aa5`, rotation "once") ran on a B200
(`submission_stage3_g_b200_seed74.json`): the fake at level 4, and a level-5 vocabulary unit that ran 1,599 s past
the deadline (training 4,408 s against 3,600 s). Its source is in the record; the file was rebuilt since.

`submission_stage3_g3.py`'s 8 B200 seeds (run `g3-seeds-2`; seeds 74 and 1 to 7) all certified, T = 64 on both
ladders with every split exact, each with its round after level 4: the round by 2,042 to 2,505 s of training,
training 2,290 to 2,790 s (records `submission_stage3_g3_b200_seed*.json`; §0 has the table). The first try at them
(run `g3-seeds`) was stopped six minutes in by the workspace's spend limit.
<!-- /OFFICIAL -->

## 6. What each change costs

**Training time.** On the official H100, the full-generic file's round came by 2,507 s of training, and it froze by
2,572 s (step 897), against sp2's 1,029 s and 1,077 s and s3i's 1,047 s and 1,092 s: about 2.4 times as long. Its
training ended at 2,770 s (sp2's: 1,277 s), inside the 3,600 s budget with 830 s to spare. On the 8 B200 seeds, the
round came by 2,042 to 2,505 s and training took 2,290 to 2,790 s, against sp2's 650 to 1,775 s and 784 to 2,010 s on
the same seeds (§0). In the tiny core (B200, the climb alone):

| step | tiny core, time to the round (after level 4 unless said) | against |
|---|---|---|
| sp2's machine (`g_base`) | 898 s | - |
| SEL.S replaced by the if block, nothing else (`g_if_brg`, `g_if1_brg`) | 1,031 s, 912 s | +15%, +2% |
| both replaced, with the four rule changes and per-root vocabularies (`g_if1_gain_root`) | 1,205 s | +34% |
| + acceptance to four digits and side-effect-free routines (`g_pure_v4`, `g_scratch_v4`) | 1,782 s, 2,183 s | +98%, +143% |
| no conditional at all, with the alias fix and deep probes of 40M (`g_ldhc_k_af_deep40`; the round after level 8) | 13,112 s | +1,360% |

Among the paths that fit the budget, acceptance to four digits and side-effect-free routines cost the most: every
problem whose cheapest programs scratch is searched one cost further, and level 3-4's probes take 2-3 times as long
(344-532 s against 152-283 s). They are what makes the climb work on the paths the
evaluator takes (§2.1-2.2), so they are not optional. The if block's vocabulary is the other cost: unrestricted (`if`)
it is 47% larger than sp2's at body cost 2, and level 5's became so large that building it took 1,599 s and up to 70
GiB of host memory (§2.1); with one item per block (`if1`) it is 7% larger. Every path that fits the budget ends at
level 4, so level 5 is a cliff to avoid, not a level to wait for.

**Size.** 244,664 bytes for `submission_stage3_g3.py`, under the 256 KiB limit because the builder drops the inlined
core's dead code (sp2's file, with it: 250,643 bytes).

**Compute.** About 26 B200-hours of tiny cores and diagnostics and about 19 GPU-hours of evaluator runs (7 of them for
the nine certifying runs), about $255 at list price, of the $400 this study was given.

**The law book did not change.** Every problem posed is sp2's book's (11 laws on 8 patterns, in every binding);
nothing was added or removed. The changes are in the instruction set and in five rules of the search that picks and
practises, each written once for every law: closeness despite aliases, the round by calls first (and its flat search
at 2M states), two concepts' deep probes a level, the "gain" rotation, each deep probe's vocabulary from its own root;
and, for the evaluator's paths, acceptance at one more size and side-effect-free routines first.

## 7. What is still designed, and what a judge may say

- The two new pieces are generic in kind, but chosen: the conditional is "if the borrow" (one flag, one polarity), and
  the shifts carry their digit through k. A judge may say the pair SHL.X SHL.A is BRG.X in two pieces. It is: two
  generic shifts compose into a two-register shift, as RCL does on a CPU, and nothing in the machine says which two to
  compose; the learner found that composition inside long division, and also used it to build a multiply no stage-3
  run had (shift-and-add over the accumulator).
- C[LDA.S] would be SEL.S in two pieces too. The search finds it (`STA.S T.lt.pair.AX C[LDA.S]`, cost 4, at level 1),
  but the learner never practises it: it writes S, and the cleaner links restore from their own input register
  (`T.sub.pair.YX C[LDA.Y]`), or, in the chains, add the modulus back (`T.lt C[T.add]`).
- The five search-rule changes and the two acceptance and routine changes were each found by watching whether the
  round came, as sp2's choosing rule was. Each holds for every law and names none, but this is more design with the
  target in view, and the write-up should be read that way. The first one has a cost on sp2's own instruction set
  (the round a level later, `g_base_af`).
- No conditional at all (LDH.C) works in principle but not in the budget. Its conditional subtraction costs 6, which
  takes deep probes seven times larger than sp2's to find. With the alias fix as well, its climb reaches the round
  after level 8, in 13,112 s of the tiny core alone (§3). No file was built for it.
- Another session, bar3/bridge ([BRIDGE.md](../../bar3/bridge/BRIDGE.md)), came at the same two instructions from
  bar 3's laws, which are stated place by place:
  - It replaced SEL.S in s3i's climb with a select at the pointer, SELC.r: A[p] := r[p] where the borrow is set. Bar
    3's learner invents it as the one-place piece of the select law. s3i's climb then finds the round at level 6, in
    1,606 s against 1,030 s with s3i's own instructions (`bar3/bridge/records/climb/inv4_h100_c1.json`).
  - It kept BRG.X, because a program that sees one place cannot move a digit between places.
  - Its analysis (no record) observed that the shift through k is exactly the one-place piece of the law x·B. Each
    place writes the digit that came up from below and passes its own digit up as the carry. libcore's SHR and BRG.X
    are not of that kind. So `SHL.X SHL.A` is the form a place-by-place law gives BRG.X.
- Nine runs of one file certify it (§0): the official H100 run and 8 B200 seeds. Its margin is time. The slowest of the
  nine trained 2,790 s of the 3,600 s budget (sp2's slowest: 2,010 s). Seed 74 took about the same time on the H100 as
  on the B200 (2,770 s and 2,790 s), so a faster GPU does not widen the margin.

## 8. Files and how to rerun

Everything is in `cpu_first/integration/generic_ops/`; nothing outside it changed.
- `libcore_g.py`: integration/lib/libcore.py with the instruction set's constants (`COND`, `SHIFT`, `BRING`), the C
  block and the shifts through k on all four machines, LDH.C, and the deadline check in `enumerate_items`. With
  `sel,zero,brg` its primitives and loops are libcore.py's, in order (`test_isa.py` asserts it).
- `spcore_g.py`: self_practice/spcore.py with the search options of §2-3, each off by default (sp2's rule):
  `alias_fit`, `calls_first`, `deep_rotate` ("once", "gain"), `deep_union`, `pure_first` (True: only A may change;
  "scratch": A and the problem's operands); `verify` was already a parameter.
- `isa.py`: libcore_g and spcore_g loaded with a chosen instruction set ("COND,SHIFT[,brg]").
- `test_isa.py`: the four machines against the reference on every instruction set, and each set's hand-written round
  on 400 T = 1 rows (§1.1). `designed_g.py`: a built file's own prediction path and round machine with the reader and
  cells at the truth and the hand-written round, on H1's splits up to T = 64.
- `core_g.py`, `modal_g.py`: the tiny core on Modal (env `ISA`, `ALIAS_FIT`, `CALLS_FIRST`, `DEEP_K`, `DEEP_ROTATE`,
  `DEEP_UNION`, `PURE_FIRST`, `VERIFY`, `ROUND`, ...; Volume `generic-ops`). `diag_gate.py`: one level's probe
  replayed from a record, with the closeness both ways (§2). `summarize_g.py`: the tiny core's table;
  `summarize_runner.py`: an evaluator record's climb.
- `stage3_g_template.py`, `make_g.py`: the stage-3 file (sp2's template, its docstring and SP_* settings), built from
  libcore_g and spcore_g with the instruction set and constants given; `make_g.py` drops the inlined core's dead code
  (s3i's curriculum, which nothing in the file calls: libcore's `Climb`, `Acceptor`, `parse`, `trace`, `round_fit`,
  `simplest`, `needs`, `task_names`, `NEEDS`), so the file stays under 256 KiB.
- `smoke_g.py`: a built file's phase C locally with toy budgets.
- `accept_odds.py`: how often a fake reduction passes acceptance at three and four digits (§2.1).
- `records/`: the tiny core's runs (`<run>.json`, `.log`); `records/runner/`: the evaluator's records.

```bash
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.generic_ops.test_isa
```

The candidate, `submission_stage3_g3.py`, is this build (it reproduces md5 `e6bd2a1f` byte for byte; `make_g.py`
refuses to overwrite that file, so a rebuild goes to another name):

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/generic_ops/make_g.py if1,k submission_stage3_g3_rebuilt.py "SP_VERIFY=((10, 1), (10, 2), (10, 3), (10, 4))" 'SP_PURE_FIRST="scratch"'
```

```bash
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/generic_ops/designed_g.py N=4 FILE=submission_stage3_g3.py ISA=if1,k
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach integration/generic_ops/modal_g.py --envs '[{"TAG": "g_if1_gain_root", "ISA": "if1,k", "ALIAS_FIT": "1", "CALLS_FIRST": "1", "DEEP_ROTATE": "gain", "DEEP_UNION": "0", "DEEP_K": "16", "ROUND": "2e6", "RULE": "lpca", "NP": "8", "DEEP_BY": "bindings", "DEEP_FOCUS": "closure", "LEVELS": "10", "SECONDS": "5400"}]'
```

g3's official run and seeds, their fetch and their verification: §0, with what a rerun needs (new run tags and
another `--out`).
