# From bar 3 to a contest-legal entry: a plan

*Written on Oct 25, 2026, while the digits study ran. Nothing in it was run and nothing was spent. Each fact cites the
write-up it comes from. The tests are proposals, and their costs are estimates from those write-ups' records.*

## In plain words

**What contest-legal means here.** The contest's 16 rules ([one-layer-deeper/README.md
§Rules](../one-layer-deeper/README.md#rules)), read strictly. The hosted evaluator has stopped taking submissions
([OVERVIEW](OVERVIEW.md) §7), so a strict reader decides, not the service. That is the project's bar 2: nothing a strict
judge would call the hand-written algorithm. Bar 3, a general algorithm learner, is not required. It is still the best
answer to rule 14's objection that a training recipe is a task-specific solver.

**Where it starts.** The self-practising stage-3 file, sp2 ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md)), learns
inside one training run, poses its own practice problems from a law book, and certifies on the official manifest. Its
variants each go one step further, and each certifies on its own (OVERVIEW §4.8). No file has them all.

**What keeps bar 3 out of the run.** No training target may be computed with Python's arithmetic (rule 14;
SELF_PRACTICE.md §2.2). Bar 3's learner computes every practice answer that way, so nothing it does moves into a run as
it is. [BRIDGE.md](bar3/bridge/BRIDGE.md) §6 names this, not speed, as what stops a run from inventing its own digit
instructions. What can move is whatever takes its answers from the data, from facts and identities, or from the
learner's own routines, as sp2's law book does.

**The order, cheapest first** (§4). Put four questions to the judge, which costs nothing. Build one file from the
variants that already certify: a tiny-core run, then one H100 run. Then carry bar 3's pieces over one at a time, the
tiny core first and certification as the gate. Through §4's five steps it comes to about $35–70 at Modal's list price,
if each step runs once.

## 1. What has to move inside the training run

Bar 3's results come from offline studies, outside any training run, and its searches run in numpy and numba on CPU
containers (BRIDGE.md §7, SCALE.md §6). Inside a run everything stays on the GPU (rule 9), where stage 3's own climb
already searches. So what moves is a rule, rebuilt on stage 3's machine, not bar 3's code.

1. **Practice answers.** Bar 3 computes them with Python's and numpy's arithmetic (SELF_PRACTICE.md §2, BRIDGE.md §6).
   Inside a run that would be hand-written arithmetic making training targets. What can move: problems built from the
   answers the data gives, and laws whose answers are facts, identities or the learner's own routines (SELF_PRACTICE.md
   §2.2).
2. **The invented digit instructions.** The bridge file takes bar 3's invented select as an instruction before training
   starts, and add-with-carry and multiply-accumulate were invented later, offline too (BRIDGE.md §5,
   [CARRY.md](bar3/bridge/CARRY.md)). A run that invented them itself would need its slices' answers from somewhere:
   Python's arithmetic is ruled out, and the model's own tables would only re-derive what it already has (BRIDGE.md §6).
   The in-run counterpart is to stop choosing the cells at all: give the climb every one-place slice of a law that lifts
   (BRIDGE.md §1), as BRIDGE.md §5 did with 61 instructions, and let it pick. That is §2, item 3.
3. **The subgoal score** ([SUBGOAL.md](bar3/scale/subgoal/SUBGOAL.md)). It asks whether the target factors through a
   problem, by a table or by one law of the book. It needs only the target's rows and the problem's answers, so it can
   move: in a run, the target's rows are the answer rows at full size, and its law test would run on the learner's own
   routines. Only the law test carries over: the table test needs answers that repeat, which small numbers have and H1's
   rows do not. Two uses:
   - *As a near-miss test for gaps that are not constants.* g2's near miss is a gap of one constant mod N, computed with
     the learned tables ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md) §3). A gap that is a factor, as the
     cube's is, is not looked for (SP_BREADTH.md §6). x³ mod N = (x² mod N)·x mod N is the score's law test exactly, and
     at tiny size a deeper search aimed by that test brought x³ in round 1 under both rules. **Cheapest test:** offline,
     the round search alone on the certified H1 run's level-4 library, as SP_BREADTH.md §4.1 did for about $1.6: with a
     near miss one law away, does the cube's round come? The known obstacle is that in the cube runs x does not survive
     the first multiply and reduction (SP_BREADTH.md §6).
   - *As the choosing rule:* §2, item 2.
4. **Bar 3's practice sources.** The two-number step brought Fibonacci ([SCALE.md](bar3/scale/SCALE.md) §4.1); the
   running sum's step and the offset are queued after the digits study. Each needs rows at chosen x: rows one apart, or
   x = 0. The contest's x are random and never 0 (SP_BREADTH.md §3), so the data almost never has such rows. Along T,
   the step is the round itself, which sp2 already finds, and g3 already checks a round against the answers of every
   one-digit T. g2's constant read off the answers plays the offset's part at full size. So these sources matter for bar
   3's breadth, not for a legal H1 entry.
5. **Scale.** Bar 3's learner counts, at x < N ≤ 7; stage 3 already works in digits at H1's size. The digits study,
   running now (`bar3/scale/digits/`), asks whether bar 3's learner can use digit loops. That is how its tiny results
   would reach H1's sizes. It is not a legality item.

## 2. What is still designed in the self-practising file

SELF_PRACTICE.md §6 lists what sp2 still has by design, and [INTEGRATION.md](integration/INTEGRATION.md) §4 marks every
hand-made part of stage 3 with the ablation's verdicts.

1. **The law book.** 11 laws on 8 patterns. A 5-law book and a book generated from a grammar both certify
   ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md) §5). The grammar needs only ingredients any such grammar
   must have: a composition of + and × that accumulates, the division law's form, and target rules from facts,
   identities and the learner's routines. Still designed: the grammar's atoms and forms, and its target rules.
   **Cheapest test:** none worth a run. This one is for the judge (§3, question 4).
2. **The choosing rule, lpca.** It rates a law on a pattern by its cheapest placement, and practises it once in every
   placement tied for cleanest. It was picked by watching whether the round came, after bar 3's rules failed: hardest
   first never gets the reduction, and of bar 3's other rules only practising everything certifies, on one seed of two
   (SP_ABLATION.md). **Cheapest test:** bar 3's subgoal score in its place, in the tiny core on two seeds. A climb that
   reaches the round takes 11 to 15 minutes on one B200, and a failing one up to about two hours (SELF_PRACTICE.md
   §3.2), at $6.25 a B200-hour. Expect little: at tiny size the score singles out nothing for x² mod N itself, only a
   tie (SUBGOAL.md). It may help only as a tie-break.
3. **The instruction set.** 32 primitives and three loop forms. The generic file replaces the two task-shaped
   instructions, SEL.S and BRG.X, with an if block and shifts that carry a digit, and it certifies using 2,770 s of the
   3,600 ([GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md)). Its digit operations are still a chosen four of the
   11 one-place slices that lift: subtract, add, multiply-accumulate and the select. The other seven are those four with
   operands swapped, an operand at zero or an output dropped (BRIDGE.md §5). With all 11 as 61 instructions, 4 times the
   climb's budgets and the conditional subtraction also probed above N, s3i's climb found every piece but the reduction.
   The reduction is in that machine at cost 5; most likely the search's cap of 50,000 goal states cut it. **Cheapest
   test:** that search without the cap, the one run the spend limit refused: one H100, at most 2 hours, so at most about
   $8 (the command is in BRIDGE.md §6). Then the 61-instruction climb with both weak checks fixed. The risk is time: it
   needed 4 times the climb's budgets.
4. **The count loop.** The learned-loop file finds the loop over T as `C.S[T.round1 STA.X]`
   ([LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md)). Still designed: the count loop instruction, whose
   stopping rule is the old countdown's; T's 2-digit field; and the answer program's search space (LEARNED_LOOP.md §7).
   **Cheapest test:** none on its own. It goes into the combined file (§4, step 2).
5. **The search's sizes and rules.** Probe budgets, the round's search (2 routine calls in sp2, 3 in g3), Occam's
   tie-break, routines and BPE (SELF_PRACTICE.md §2.4, §6). The s3i ablation found routines needed, BPE speed only, and
   Occam's tie-break not needed once the weak checks are fixed (INTEGRATION.md §4, OVERVIEW §4.7). **Cheapest test:**
   none proposed. They are sizes, and the judge rules on them with the rest.
6. **The reader and writer.** Stage 1's grammar: 24 reading hypotheses, 3 pointer shapes and the run rules, with
   designed logic choosing a prompt's hypothesis (INTEGRATION.md §4; [REVIEW.md](REVIEW.md) §5, rule 8(d)). It reads
   exactly three fields (SP_BREADTH.md §6). **Cheapest test:** none small. A reader for any number of fields is a
   redesign.
7. **The cells and their lessons.** Three tables whose input shapes are designed, each learned from laws (the
   subtraction from the order of the numbers), and the mined facts' grammar. The late table lessons are not needed
   (INTEGRATION.md §4). **Cheapest test:** none proposed.
8. **The schedule.** Phases A–D, and a freeze that sets the learning rate to 0 once every check has held for 200 steps.
   The strict builds took out the clock ([STRICT.md](integration/strict/STRICT.md)), but the freeze is still early
   stopping on the model's own checks (rule 4). **Cheapest test:** one official H100 run with the freeze removed,
   training to the evaluator's own deadline. If the tables stay exact, the freeze can go. About $4.

## 3. Questions only a judge can settle

REVIEW.md §5 states each one with the lines to read. Ask them first: if question 1 goes against us, the whole stage-3
line is out, and nothing above matters.

1. **The climb is a program search inside the training forward pass** (rules 8 and 14), its result kept in buffers in
   the strict builds. The rules call "parameter-free work" valid and allow iterative mechanisms (rule 4, and the note
   after the rules). Is a search whose result carries from step to step "hidden training work"?
2. **Training returns zero logits** (rule 8(a)), and the damped gradient underflows to 0 on the deepest paths at a
   perfect prediction (8(b)). The batch's answer rows would fit through the real path, at 200 to 400 s a run, but would
   teach nothing once the round reproduces them; the whole batch does not fit in an H100 (STRICT.md §5).
3. **The model keeps rows and its own schedule** (rule 4): a pool of read rows for practice problems and of answered
   rows for the round's search, kept as buffers in the strict builds; phases A–D; the freeze (§2, item 8).
4. **The instruction set and the law book were chosen knowing the task family** (rule 7). §2 shrinks both. The judge
   says when what is left counts as general.

## 4. The order, cheapest first

1. **The four questions.** Free. Everything else waits on question 1.
2. **One file with every variant that certifies**: the fixed checks, the strict build, the learned loop, the generated
   book and the generic instructions. Each certifies alone (OVERVIEW §4.8); together they remove the most at once, and
   they may interact. The risk is time: on the official H100 the generated book trained 2,428 s and the generic
   instructions 2,770 s of the 3,600. The bridge's select stays out: it was invented before training, and the if block
   does its job inside the run. **Test:** the combined climb in the tiny core (one B200), then one official H100 run,
   about $1.5–2 if it certifies and about $4 if it uses up the hour (SP_BREADTH.md §6). The tiny core trains nothing and
   its times do not carry over to the evaluator (GENERIC_OPS.md §2.1, §3), so the H100 run is the real test. If it does
   not fit, which piece costs the time is the finding.
3. **The uncapped reduce search** (§2, item 3; at most about $8). If it finds the reduction: the 61-instruction climb in
   the tiny core, then in the combined file.
4. **The subgoal score**: first as the cube's near-miss test (offline, about $1.6), then as the choosing rule (the tiny
   core, two seeds).
5. **The freeze** (§2, item 8; about $4).

The digits study and the two-source study, both already approved, serve bar 3's scale and breadth, not this path.
