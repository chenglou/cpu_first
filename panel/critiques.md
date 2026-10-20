# Adversarial critiques (one per design, in completion order)

# Verdict: fatal  (p certify = 0.04)

## verdict

fatal


## probability_certifies_T64_both_profiles

0.04


## fatal_flaws

- Rules spirit (rules 7 and 14): the forward pass is a hand-wired Montgomery modular-exponentiation netlist. Horner D2B, 3N, the radix-4 entry, carry-save squaring, Walter's bound, the predicated counter, the halving exit, double-dabble and the gather are all designed. The training labels decide only a 10-row nibble-to-digit identity readout. The shared gate is learned only nominally: by enumeration, the axiom loss has exactly one zero, the full adder (1 of 65,536 tables, with real operands at W=28 and NOT built from the cell). So the loss is just a roundabout way of hard-coding the 16 bits. No version of this lens fits the rules' spirit, and the design concedes this itself.
- The calibration recipe fails as written. Step 1 builds the axiom operands with a hard D2B made from the current best candidate, so the hypothesis being certified also produces the data that certifies it. Measured with the design's own losses at W=28 on H1 rows: 0 of 8 populations (0 of 128 candidates) reached the exact FA in 600-1,000 steps. A typical run: the first best candidate's D2B emits all-zero operands, and every candidate's axiom loss falls to 0.004-0.06. That is lower than the 0.015-0.033 of truly exact candidates in the clean-operand control, so the argmin/EMA selector cannot tell it has failed. Even a population whose first operands were informative (60% ones, 96% distinct rows) had collapsed by step 20. Enumeration explains it: 6,098 wrong tables meet all four axioms exactly on operands from their own D2B, 13.4% of random tables emit all-zero operands, and 21.4% emit constant ones. 'Garbage bits are fine' is false. The flaw can be fixed (see fixes), but the recipe as specified never calibrates.


## serious_issues

- The obvious parameter-free fix makes calibration unreliable at P=16. Using the prompt's own BCD bit strings as operands keeps uniqueness (1 of 65,536 tables survives all four axioms), but only 3 of 96 candidates (about 3%) became exact, and 3 of 6 populations had none. A new attractor with 3-5 wrong entries (loss about 0.74) dominates, next to the known 13-wrong one (about 1.3). The design's 'about 50% per seed' holds only for exact binary operands (18 of 32 here, first locks at 60-584 steps), and those require an already-correct gate. So P=16 is about a coin flip. P of 256-512 is needed (failure about 4e-4 or 2e-7, if candidates are independent).
- The saturation shield is ineffective under Adam. Axiom gradients and main-loss gradients reach a logit through the same factor sigmoid'(logit), so multiplying logits by 4 leaves their ratio unchanged, and Adam's step size does not depend on gradient scale. At |logit|=12 the gradient is about 1e-6, far above eps=1e-8. The only protection is distance: about 240 consistent steps at lr 0.05 to flip a bit. A certified gate must be frozen (lr 0, or a 1e-6 in-graph gradient scale), not saturated.
- Rule 8: the truncated backward through the core passes the letter only if it is done with an in-graph gradient-scaling op. Done with detach or no_grad, about 1,900 levels of input-dependent computation sit outside the autograd graph. Either way, the core's gate parameters get essentially no task gradient. Hard thresholding in eval, the fixed token-to-BCD embedding (token-7, marker IDs) and the in-forward population argmin with an EMA buffer are further gray points.
- Before lock-in, the loop bound 'max decoded T' comes from a wrong gate and can reach 127 squarings, about 12,100 FA levels per step instead of about 1,930. The main path should be skipped or clamped until a candidate is certified.
- The 'needle universality' and triangularity claims ('labels can check but cannot teach' heavily reused cells, 'for any reduction algorithm') rest on 4 random-init runs of one multilinear relaxation at 8 bits, run for 800 steps. E2's own near-init arms succeeded 2 of 4. This is a hypothesis, not a result, and it is not needed for this design anyway.
- Implementation surface: the netlist is large (about 1,930 levels at T=16 and 5,839 at T=64), and every block must be unit-tested against the integer oracle. My gate-level rebuild passed on the first full run, so the risk is moderate, but a submission would need the same test discipline.


## fixes

- Separate operand generation from the hypothesis being certified. Feed the axioms parameter-free, prompt-derived bit strings (the BCD of N and x, which keeps uniqueness: 1 of 65,536). The alternative is synthetic random bits: about 50% per candidate, but grayer under the no-augmentation rule. Never use D2B output from the candidate being certified.
- Raise the population to P=256-512. It costs nothing extra in kernel launches on an H100, and it absorbs the about 3% per-candidate success rate with BCD operands. Once one candidate is certified, optionally restart the rest on its exact-binary D2B operands.
- Select by hard axiom error (thresholded table, exact check), which is a true certificate because the solution is unique, and use the soft EMA only to break ties. Add an operand-diversity guard: refuse to certify on a batch with fewer than k distinct operands, or when the fraction of ones is near 0 or 1.
- Freeze a certified gate (lr 0 for that group, or route the main-loss path through a tiny in-graph gradient scale) instead of multiplying its logits by 4. Skip or clamp the main path until certification. Train U only after that; it converges in tens of steps from about 2,800 labeled digits per batch.
- Do the backward truncation with an in-graph gradient-scale op, not detach or no_grad, if rule 8's letter matters. Run the hard eval path in fp32, or rely on 0/1 values being exact in bf16.
- On the rules: no fix exists within this lens. Present it as an exactness and compute reference ceiling. To move toward the spirit, the wiring itself has to be learned, for example by MoE selection over designed datapath variants driven by labels, or a learned micro-program on a generic bit-slice ALU.


## hidden_assumptions

- The operand source is harmless ('the axioms are identities over all bit strings'). In fact, operand diversity is what turns the axioms into a certificate, and the recipe lets a wrong gate pick its own operands.
- The 12-bit uniform-random calibration rate (about 50%) carries over to the operands actually available in training. It is 0% with D2B feedback and about 3% with BCD; about 56% only with exact binary operands, which presuppose the answer.
- The 16 candidates fail independently, so P(all fail) = 0.5^16. With shared operand feedback they fail together: all 16 collapse in the same step.
- A low soft axiom loss means calibrated. Collapsed populations score lower losses than genuinely exact candidates.
- Multiplying logits by 4 attenuates destructive updates under Adam (it does not).
- Truncating the backward still counts as an 'unbroken gradient path' under rule 8.
- The digit and marker token semantics (value = token-7; IDs 2/3/4) count as 'formatting', not a hard-coded embedding.
- The evaluator would accept a fixed task-specific netlist with a learned gate as 'the learned model' (rules 7 and 14 say no).


## algorithm_check

All scripts and outputs are in /private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/panel/critique-binary-cpu/.

(1) The design's simulations reproduce. sim_pipeline gives 10,752/10,752, and sim_variants gives 5,376/5,376 per profile as mixed-T predicated batches. Highest carry-save bit 24, S <= 2.251N, s <= 1.294N, exit < N. The theory agrees: S < 3N inside a squaring; Walter's bound gives s < 2N with R = 2^26 > 4N for every N < 2^24; the exit is < N because s is never 0 mod N; N^-1 ≡ N mod 8.

(2) gate_sim.py is my independent gate-level rebuild, in which every gate is the single FA table: AND, OR, XOR, NOT and MUX as FA instances, carry-save Horner D2B with a Kogge-Stone adder, 3N, the radix-4 entry with borrow-out selection, predicated carry-save Montgomery with a 7-bit FA down-counter, the 26-halving exit, and FA-built double-dabble with right-aligned readout. Result: 5,376/5,376 on both profiles (all 10,752 certification prompts, T = 1..64 in one batch). Measured critical path: D2B 35 levels, entry ends at 260, +92 per squaring (84 for the first), exit 80, B2D 115. That is 5,839 FA levels at T=64 and about 1,930 at T=16, slightly under the claimed about 6,450. Compute is fine: roughly 234k levels for the whole eval suite, seconds to about a minute launch-bound, and training at about 0.1-0.3 s per step, so 3k steps fit easily.

(3) Uniqueness enumeration (enum_real.py, enum_bcd.py) with real binary N and x at W=28, NOT from the cell. Counting tables that pass the cumulative set of axioms: round-trip alone 796, plus commutativity 264, plus A+0=A 2 (the FA and a carry-passthrough table), plus place value 1 (the FA). With BCD operands, all four axioms also leave 1. With all-zero operands, 11,264 pass.

(4) Calibration dynamics (calib_pop*.py), using the design's losses, Adam at 0.05, cosine schedule, clip at 1, P=16 per population:
- Exact binary operands: 12/16 and 6/16 exact; argmin picks an exact candidate.
- Literal D2B-from-best recipe: 0/16 in all 8 populations, with populations collapsing to all-zero operands and near-zero loss.
- BCD operands: 3/96 candidates exact; 3 of 6 populations have none.
- enum_selfconsistent.py: 6,098 wrong tables are exactly self-consistent on operands from their own D2B.

Technical outlook if the spirit condition is waived: with decoupled operands, P >= 256 and hard-certificate selection, about 0.85-0.9. As literally specified, about 0. In the rules' spirit, about 0, hence the 0.04.


## best_idea_to_keep

Two ideas transfer to any lens. First, learning by speculation plus certification: train many cheap candidates of a shared cell in parallel on shallow contexts, then select with an exact, label-free test whose solution is provably unique (here the four axioms, 1 of 65,536). That turns an optimizer succeeding 3-56% of the time per candidate into a reliable one, as long as the certificate's operands do not come from the hypothesis under test. Second, the verified constant-free binary Montgomery datapath: a fixed R=2^26 with Walter's bound covers every N < 2^24, the quotient bit is a parity bit (N^-1 ≡ N mod 8, so radix up to 8 needs no per-N constant), the exit is comparison-free, and the only top-down work is a per-example 13-step radix-4 speculative entry. The per-step datapath is therefore LSB-first with zero per-modulus constants, which makes it a good exact reference that a learned design can be tested against.



---

# Verdict: fatal  (p certify = 0.1)

## verdict

fatal


## probability_certifies_T64_both_profiles

0.1


## fatal_flaws

- Rules 7 and 14 (no hard-coded algorithm, no task-specific solver). The forward pass is a hand-written radix-10 Montgomery program. That covers I/O parsing, the 9-step domain entry with carry-out select, the MonPro schedule, the quotient rule m = t0*N' mod 10, carry-save routing, the resolve ripple, the MonPro(s,1) exit, and the decrement-by-99 counter with its halting mux. The learned parameters only fill lookup tables that this fixed program calls. Every designed piece is load-bearing: remove any of them and the solution goes with it. No variant of this design is compliant.
- Rules 6 and 14 (no hard-coded weights, no data augmentation). The table contents do not come from the task. Phase 0 runs about 1,500 evaluator steps on designer-generated arithmetic and ignores the evaluator's batches. The targets are computed in Python: (a+1)%10, Peano recursion, commutativity, MUL recursion, and 9's complement. The unique fixed point of these targets is the true tables, so the loss is an answer key, which amounts to torch.load with extra steps. I measured 18/18 seeds at both init scales (0.01 and 1.0) reaching 100% exact FA, MUL and COMP in 400 Adam steps, 0.5 s on CPU. Information learned from evaluator labels: N' (4 choices of 10, about 13 bits) plus a 10->17 output map (about 41 bits). The table content is about 1,560 bits (log2 of 20^200 * 100^100 * 10^10), and all of it comes from the designer's loss. Even N' is fixed by the learned MUL table (the unique m with m*N0 = 9 mod 10), so data-learned content is near zero by construction.


## serious_issues

- Rule 8 as written. Eval uses argmax tables, integer gathers and an argmax-collapsed gate. That is input-dependent computation outside the autograd graph, with no gradient path from the loss to the parameters that decide the prediction (the tables and the N' gate).
- The recipe cannot be implemented as stated. The model never sees labels, so it cannot return 'speculative log-likelihoods' in aux. It must return per-candidate logits, and token_training_loss must form the mixture. Similarly, 'stop via max_steps when T=16 train accuracy is 100%' is impossible because max_steps is fixed at submission. Phase switching on forward-call count also drifts if multi-pass or batch reuse is enabled.
- Phase 1 and Phase 2 table updates carry risk and no benefit. Early in Phase 1 the posterior over the 10 N' machines is roughly uniform, so about 90% of the table gradient comes from scrambled machines. Adam's per-parameter normalisation turns those tiny, biased straight-through gradients into lr-sized steps; the design's own probe went from 2 to 11 wrong entries at lr 1e-2. Phase 2 has nothing to gain, because the machine is already exact after Phases 0 and 1. The 16 replicas all converge to the same unique fixed point, so they add only cost.
- The central finding is overgeneralised. 'Mod-N scrambles the output after any single wrong entry' is partly a Montgomery artifact: an error d enters the result as d*R^-1 mod N. Same generic tables, same 1,000 T=1 rows, one wrong entry: rows that fail keep 0.126 digit accuracy under Montgomery versus 0.293 under top-down (schoolbook square plus MSB-first long division). Pass rate is 0.42 versus 0.68. Both are at chance by about 16 wrong entries and with random tables. So 'random-init, label-only, free-table learning is dead' survives. 'Any T=64 system must import arithmetic by design' is not established: structured parametrisations and architectures with a non-chaotic soft relaxation were never tested.
- The decisive experiment does not decide this design. CADET-M certifies, or doesn't, whatever w* turns out to be; its fate rests on how the rules are read, which no experiment settles. Part (c) is guaranteed by the integer simulation. Parts (a) and (b) inform the panel, not this proposal.
- A hidden-knowledge hinge. The fixed input map (token 7+d -> one-hot d) is treated as plumbing, yet the self-test anchor (successor = token order) depends on it. It is task semantics hard-wired twice: once in the input wiring and once in the loss.


## fixes

- For a compliant entry, no fix inside this design exists. Repurpose CADET-M as an exact reference oracle and test harness: validate learned designs against it and use it to analyse per-step error offline. Do not feed its intermediates into training, which would be augmentation.
- Shrink the designed arithmetic. Replace the answer-key self-tests with law-only consistency losses: identity, commutativity, associativity of ripple addition, distributivity of MUL over FA, with no successor facts. Then let the labels pick the survivor among the residual solutions by output-level speculation. First measure how large that residual family is. If it is at most about 100 per independent factor, labels can select it, as the N' result shows (100% vs at most 0.5%).
- Learn the microcode as well as the tables. Offer a small menu of complete schedules (operand routing, quotient rule, entry/exit variants) and select them with the same likelihood mixture over whole machines. That is the 'temporarily weak connections between machines' idea, used where it actually carries information.
- If built anyway, apply mechanical compliance and stability fixes. Keep the straight-through one-hot path at eval: forward values are identical, the kernel count roughly triples, and eval still takes seconds to under a minute. Return per-candidate logits in aux. Set table LR to 0, or detach the tables, after Phase 0. Drop Phase 2 and the replicas. Use a fixed max_steps of about 2-3k.


## hidden_assumptions

- That gradient descent on a designer-written loss with a unique, known solution counts as 'learned from random init'. It satisfies rule 6 in letter only.
- That about 1,500 steps of training on synthetic arithmetic, ignoring the evaluator's batches, is not data augmentation.
- That a fixed, ordered digit-token one-hot is plumbing. In fact it supplies the successor function the whole Phase 0 rests on.
- That integer-gather eval and an argmax-collapsed gate satisfy the rule-8 gradient-path requirement.
- That the model can compute label likelihoods in forward, and that max_steps can depend on training accuracy.
- That N' needs to be learned from labels at all. It is determined by the MUL table, so the 'learned from data' content is cosmetic.
- That results for free decimal tables in one datapath (Montgomery, the worst case for credit) generalise to every architecture and parametrisation.
- That table coverage in the training data matters for exactness. It does not: the tables come from self-tests, so the 6.8k-visit coverage argument is irrelevant to certification.


## algorithm_check

I wrote an independent vectorised numpy version of the full digit-level datapath from the design text, without importing their sim (panel/critique-decimal-machine/vsim.py). It gives 0 mismatches on all 10,752 certification prompts: 14 rungs, both profiles, T=1..64, including 8-digit OOD N. It also gives 0 mismatches on 4,964 training rows (all 2,000 rows with T<=2 plus 3% of rows with T>=4). Every invariant held: S2_0 = 0 at every iteration; carries stay binary, which is trivially closed since 9+9+1 <= 19; the entry carry pattern is monotone with a unique select; top carry is 0; resolve carry is 0; s~ stays below 1.0104N; and the counter runs exactly T steps. FA lookups at certification total 74,763,264, matching the claimed 74.8M. All 200 FA and 100 MUL entries are hit at certification, and all appear in the training sample. Walter's bound checks out by hand: s~ < 2N and R = 10^9 > 4N give MonPro < 1.07N, and the exit value is below N + 2N/R and cannot equal N because x is a unit, so no final subtraction is needed. N' check (nprime_check.py): in each N mod 10 class exactly one candidate gives 100% and the rest give at most 0.5%. Self-tests (selftest_seeds.py): 18/18 seeds reach exact tables in 400 steps. Label signal (perturb.py, 1,000 T=1 rows): with a single wrong entry the median pass rate is 0.39-0.47 (COMP 0.02), and rows that fail have chance-level digit accuracy (0.11-0.13). Pass rate for w wrong entries is 0.37 / 0.18 / 0.03 / 0.0014 / 0 at w = 1 / 2 / 4 / 8 / 16. Top-down control (topdown.py) degrades more slowly but is also at chance by w = 16. Compute: training needs at most 1,250 levels at T=16, which fits easily. Eval runs about 2.1k MonPros over about 41k prompts in batches of 512, which takes seconds and in any case under a minute. Conclusion: the arithmetic is exactly right, and if rules are ignored this would certify T=64 on both profiles with about 0.95 probability. The 0.1 estimate is roughly 0.95 times the low chance (about 0.1) that a strict reader accepts designed microcode plus answer-key self-tests as a learned model. Inside the rules spirit, the remaining label-only version has essentially no chance.


## best_idea_to_keep

Use labels only to select among small menus of complete alternative machines, through a likelihood mixture at the output; the N' separation was 100% vs at most 0.5%. This pairs with the measured fact that label signal disappears once about 8-16 table entries are wrong. Keep a generic shared cell (one decimal FA plus one MUL) for exact width and T extrapolation, with the loop counter as a decrement-by-99 through the same FA. Keep the exact integer machine as a verification oracle.



---

# Verdict: fatal  (p certify = 0.3)

## verdict

fatal


## probability_certifies_T64_both_profiles

0.3


## fatal_flaws

- Rules spirit: the whole residue computation is hand-written in the forward pass. That covers token→value decoding, the parse family, the anti-diagonal square, the fold through a per-N power table, telescoping-carry semantics, candidate enumeration S−qN, the MSB-sign rule, the boundary-select formula and the count formula c=a·tens+b·ones. What is learned is cosmetic: slopes ≈0.1/0.01/0.001, (a,b)≈(10,1), an offset-0 pointer, an identity token map, a 2-state borrow FSM and a 3-state sign FSM. Rule 7 says outputs must be produced by the learned model; here the learned part only formats an answer the designed engine already computed. The design itself concedes this. The fallbacks it needs (an exact-floor carry to avoid overflow, a label-side oracle to bootstrap selection) add more hand-written logic. On the brief's research question (how much of the datapath can be learned) the design answers 'none of it', so it is a hand-coded baseline, not a candidate. Technically it is fixable and would likely certify after the fixes below; the disqualification is about the rules spirit.


## serious_issues

- The OOD profile fails as specified. The setup recurrence P_{m+1}=10P_m−q̂_m·N needs q̂≈10P_m/N, so it must read N's leading digits. That contradicts the claim that 'nothing is conditioned on N's leading digits'. Coverage: 370/505 OOD moduli lead with 1 or 2 (training has 2/430 and 0/430), and 471/505 OOD moduli have a top-3 prefix never seen in training. Test: I trained an MLP on (top digits of 10P_m, top 3 digits of N) by direct regression, which is stronger than the design's label-free objective, using training moduli only. Result: 0/768 wrong on in-distribution T=64, but 480/768 wrong on OOD T=64, with 508/768 rows losing fp32 exactness and |S|/N reaching 1e14. Mean quotient error was 2.46 for leading digit 1 and 1.44 for leading digit 2, against about 0.4 on training moduli.
- The magnitude objective Σrelu(|digit|−5)² + λ·relu(|col|−2^20) cannot see how large a value is. A non-canonical P_m with all digits ≤5 costs nothing until it overflows the window. With a systematic 15% under-estimate (γ=0.85), |P_m|/N reached 8.3e3 with zero penalty and no window overflow. So the quotient readouts get essentially no gradient across a wide band, and 'learned magnitude control' is not actually trained by the stated objective.
- Training the 'transparent' output stage by plain mixture CE did not bootstrap when I replicated it. Setup: exact core states, 6,500 training rows; learned count decoder, sign FSM, canonicalizer, writer and token map; designed boundary selector. With the true c given and only q latent, loss stayed at chance (2.22 vs ln10=2.30) and exact match at 0.1% after 1,500 steps. Fully joint was 0% after 1,000 steps, with a drifting away from 10. With the oracle (c*,q*) it learns: 77% at 750 steps at lr 3e-3 and still rising. At lr 3e-2 even the oracle run collapsed the writer onto a wrong offset. So the label-side oracle is required, not an 'optional accelerator'. Caveat: single seeds, short CPU runs.
- The claim that sandboxed units 'cannot cause a wrong answer' is false in three places. (1) The output-stage carry rounds are tagged [Ls], but they decide whether |d|≤9, which the MSB sign test needs, so they are correctness-critical. (2) Setup quotient errors cause window misses or fp32 overflow, as above. (3) With carries inert or wrong-signed at random init, digits grow roughly 9 → 1e5–1e7 → 1e15 → inf within 3–4 squarings. Every batch contains T=16 rows, so the forward pass yields inf or NaN, and one NaN step corrupts Adam. The guard the design proposes is a hand-coded floor carry, which makes the learned carries redundant.
- The noise model behind the robustness claims is independent, zero-mean, per-decision noise. Trained units make systematic errors that depend on the input (for example on N's prefix). My γ sweep tolerates only about −15%/+20% systematic quotient bias; at γ=1.2 the final q reaches 116, near the ±128 window. Random ±50% noise gives 59 + 104 wrong out of 1,536.
- Count-decoder extrapolation is fragile. Training constrains |β|<1/16 and |α+6β|<1/2, where α=a−10 and β=b−1. T=32 and T=64 need |6α+4β|<1/2, which holds on only 1/6 of the region training allows. It works only if the soft p(c|T) is kept until (a,b) are centred, not hardened early.
- Reader edge cases need designed masks. Under token−7 the T marker reads −3, so a 1-digit T needs tens=0 and x/N slots beyond their fields need 0. The 576-hypothesis family does not express field boundaries, so the parse is designed in practice.
- Rule 8 gray area: the carry and quotient parameters shape the representation of S that the output stage reads, but they are detached from the CE path. Their only gradient is the aux term.
- Compute is optimistic but fits. The 16×65×4096×13 candidate mixture with FSM scans means GB-scale activations, so expect roughly 50–200 ms per step rather than 15–40 ms. That still allows more than 10k steps, and eval at T=64 is trivial.


## fixes

- Make the setup exact and independent of N's leading digits, applying the design's own rule that width replaces prediction. For each m, enumerate P_{m+1}=10P_m−qN for q∈{0..9}, or a small window around a cheap estimate, and pick the answer with the same sign FSM and boundary selector as the output stage. These are label-trained and shared. Or use a designed exact quotient. Either removes the 1xx/2xx coverage hole.
- If any learned quotient remains, penalise value size, not digit size: put weight on nonzero digits at positions ≥k in P_m and in S. Also widen the eval window to ±1024 or more; that is cheap, about 2k candidates × 13 digits × 768 rows.
- Promote the loss-side oracle (c*,q*) to required auxiliary supervision. Use CE on the count index and the selected q, plus per-candidate sign targets (y_q≥0 ⇔ q≤q*). Keep the per-digit mixture as the main loss. Use lr ≈3e-3 for the transparent tables (3e-2 collapsed the writer) or an entropy floor on the writer pointer.
- Guard against NaN at init: clamp or nan_to_num the state, and warm up the carry slopes on setup tensors before running the 16-step loop. If an exact-floor fallback is kept, say plainly that it is designed.
- Keep p(c|T) soft (Gaussian kernel) until convergence and add an oracle-supervised (μ−c*)² term. Before stopping, assert |6(a−10)+4(b−1)|<0.5.
- Add explicit field masks to the reader so the tens digit is 0 for 1-digit T and slots outside a field read 0.
- Treat the output-stage carries as correctness-critical. In exact simulation, 2 rounds of 3-level carries normalise every candidate in a ±128 window.
- Rules spirit, for the research question: use this engine as an exact reference or teacher. Then replace one designed piece at a time with a learned module placed transparently, for example letting the shared sign/select FSM drive the setup reductions, and measure what still learns. Do not present the engine itself as the learned system.


## hidden_assumptions

- That the setup quotient can be learned without reading N's leading digits, or that a learned estimator extrapolates from training prefixes (mostly 3xx–9xx) to OOD prefixes (73% are 1xx–2xx).
- That a digit-magnitude penalty trains value-size control.
- That the per-digit mixture over (c,q) makes a randomly initialised sign FSM, selector, count decoder and writer learnable without label-side search.
- That the forward pass stays finite at random init while the carries are untrained.
- That independent zero-mean noise injection is a fair stand-in for the errors of a trained unit.
- That (a,b) land at the centre of the region training allows, not anywhere inside it.
- That rule 7 allows a parameter-free modular-arithmetic engine as long as small trims are learned.
- That the evaluator's bf16 autocast can be switched off for the core. It can, and it must be: bf16 represents integers exactly only up to 256, while column sums reach about 1–8.6k. TF32 is actually harmless here because the fold's inputs stay below 2048.
- That output-stage carries are sandboxed. They decide the sign test.


## algorithm_check

I wrote my own vectorised exact-integer simulation of the design as written: square, fold, 2 rounds of 3-level telescoping carries, fold, 2 rounds; then the output stage (S−qN for q in ±128, a fixed carry budget, MSB-sign test, boundary select).

**Core with exact P (balanced digits), all 1,536 T=64 certification rows**
- 0 wrong, and no intermediate reaches 2^24.
- max|col|: 1,213 in-distribution, 1,402 OOD. The design's ≤8.6k figure used canonical [0,9] P digits.
- |digit| ≤ 6; |S|/N ≤ 5.5 / 9.1; final q in [−6,5] / [−10,7].
- Still exact with ±10% carry-slope error and 2 rounds per fold.
- The output stage needs only 2 carry rounds; the sign test and select are exact.
- The core claims hold.

**Setup with learned-style quotients**
- Zero-mean ±30% per-step noise: 0 wrong, final q up to 74. |P_m|/N reaches 9e4, which is harmless only because the columns C_m that multiply those P_m are zero.
- ±50% noise: 59 + 104 wrong.
- Systematic bias: γ from 0.85 to 1.2 is tolerated (final q up to 116 at γ=1.2).
- MLP quotient estimator, directly regressed on training moduli: in-distribution 0/768 wrong; OOD 480/768 wrong, 508 rows fp32-inexact.

**Data**
- Leading digit of N, train: {1:2, 3:18, …, 9:19}; OOD: {1:204, 2:166, 3:49, 8:36, 9:50}.
- 471/505 OOD moduli have a top-3 prefix never seen in training.

**Output-stage learnability** (exact prefix states, 16 prefixes × 21 candidates)
- Hand-set true parameters give 100%, so the harness is correct.
- Oracle (c*,q*): 77% at 750 steps, lr 3e-3.
- True c given, q latent: chance after 1,500 steps.
- Fully joint: 0% after 1,000 steps.
- Single seeds on CPU, so this is suggestive rather than conclusive.

**Count decoder**: only 1/6 of the region training allows extrapolates to T=64 (analytic).

Total compute was about 9 minutes, over the ~5-minute target; most of it was one 300 s joint run.

Scripts, in /private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/panel/critique-gpu-native/:
- stats.py
- core_sim.py
- setup_sim.py
- mlp_quot.py
- make_states.py
- learn_output.py, learn_output2.py, learn_output3.py
- check_hand.py

How to read the probability (0.3): it is the technical chance that this design, built competently with its own listed fallbacks, certifies T=64 on both profiles. About 0.75 with the fixes above; about 0 under a strict rules-spirit review.


## best_idea_to_keep

Keep the split between sandboxed and transparent learning. A decision made through value-preserving integer moves (carry amounts, quotient guesses) can only change magnitude. That costs window and headroom, never the residue. Every residue-setting decision should therefore be an exact select among enumerated candidates, trained on the selected answer. Apply this consistently, including in the setup, so that width replaces prediction and nothing depends on N's leading digits. The substrate worth keeping is carry-save folding: no comparisons, about 7 ops per step, columns ≤ about 2k with balanced P digits, exact in fp32 (even TF32), verified exact on all T=64 certification rows. It also makes all 102k rows clean supervision. Use it as the exact reference or teacher, and hold the learned system to the transparency test.



---

# Verdict: fatal  (p certify = 0.04)

## verdict

fatal


## probability_certifies_T64_both_profiles

0.04


## fatal_flaws

- Rules spirit and letter. The forward pass hard-codes the whole modular-squaring algorithm: token parse, exact column ALU, SRT micro-program, carry transfers, place-value readout, candidate exit, loop wiring and output placement. What is learned is trivial: a 2-D quotient estimator, a carry table, a 3-way select, a decrement table and an identity unembedding. That breaks rule 7. The loop runs under torch.no_grad and every decision is an argmax. Rule 8 says all input-dependent computation must be inside the autograd graph, with an unbroken gradient path to the parameters that make the prediction. This design breaks the letter of that rule.
- The 'label-derived hindsight target' does not use the label at all. With rep = y if cur>=0 else y-N, the target jr = (cur-rep)/N equals trunc(cur/N) exactly, and trunc(jr/10^i) equals trunc(cur/(N*10^i)). I checked this on 2.4M cases with 0 mismatches. So token_training_loss contains an exact multi-precision long divider, and P is distilled from it. That is a task-specific solver in the loss (rule 14). The 'bootstraps from 1000 T=1 labels' story is also wrong: the labels only gate rows through the integrality check, and P could be trained on any inputs at all.
- The exit as specified fails every certification rung. It runs a 12-step division with the shared P, then selects among {r-N, r, r+N}. That only covers post-division remainders in [-N, 2N). The trunc targets leave negative states in (-N, 0], so a P that is slightly off at |cur| = N lands at about -1.01N to -1.07N. I trained P with the design's own on-policy+exploration script for 6000 steps; this is the setting where in-loop T=64 stays contained on both profiles. The learned exit still failed 1-14 of 768 rows on every one of the 14 rungs, T=1 included, and every failure had r < -N.
- The learned counter cannot be identified from training. The digits 3, 5, 7 and 9 never appear in any training T, and one-hot tables with i.i.d. init and Adam are permutation-equivariant. So swapping those four digits is an exact symmetry of the training loss, and D(3) is not determined by the data. I fixed Z and B to their true values and enumerated D. 432 tables reproduce every training halting time. Only 6 of them (1.4%) get T=32 right and 24 (5.6%) get T=64 right. Among the 24, the level of '3' in the countdown is uniform over {3,5,7,9}, so P(T=32 correct | T=64 correct) is 1/4. The rungs must certify consecutively, so this alone caps the chance of certifying at 25% or less. The 8-copy population gate cannot help, because no training signal separates copies that differ only on T=32.


## serious_issues

- The evaluator-compatible (on-policy + exit exploration) recipe is very sensitive to the step count. I ran the design's own script unchanged on CPU with 256 T=1 rows per step. At 1500 steps, 0 of 8 seeds (hind and hind_jit) kept T=64 contained; all overflowed on both profiles. At 2500 steps, 0 of 2 did; the state blew up by T=2-16 and reached |r|/N up to 6e9. At 6000 steps, 2 of 2 did, with max |r|/N of 2.0-2.95. The '~500 steps' figure belongs to the teacher-forced variant, which also worked for me (2/2 at 1500 steps). But teacher forcing needs either a nested P call inside the loss (rule 14) or an exact divider inside the forward pass (rule 7).
- The slack argument was measured around round(w/N), but the learned P is trained toward trunc(w/N), so there is less slack. With a trunc oracle and digit set [-19,19]: +-1 errors at 30% stay safe (max |r|/N 2.0). +-2 errors at 10% overflow the training-pool T=64 profile. +-3 errors at 0.1% per decision overflow both profiles. Each profile has about 983k decisions, so P's +-3 error rate must be well below 1e-6 on in-loop windows that are only ever supervised through exit windows.
- One shared P has two jobs that pull against each other. In the loop it needs margin: jitter of +-0.1 to 0.15 in log|u| is kept for OOD robustness. At the last exit step its decision boundary at integer |cur/N| must be exact. The jitter flattens the cross-entropy around exactly the boundaries the exit depends on. My exit failures (0.1-1.8% per rung) are this effect.
- Counter trainability is untested and looks poor. I trained soft one-hot D/B/Z tables with the halting NLL across 64 seeds, assuming a perfect exit, so the loss reduces to -log w_T. They fit T=1, 2 and 4 (61-64 of 64 seeds) but T=8 in only 8/64 and T=16 in 0/64. Most seeds ended in a cycle that never halts.
- The gradient clip is neutralised by multiplying the loss by 1e-3. Rule 4 says gradient clipping is owned by the evaluator, so deliberately making it inert is a gray area.
- The feature choice (log-domain features plus jitter) was validated on 983k windows taken from the depth_ood_n_t_64 certification profile. That is tuning on the test set (data inspection). The leading-digit shift could have been derived from the bit sizes alone, but it was not.
- The hindsight scripts compute jr in float64. Values with W=22 reach about 1e22, which is beyond 2^53, so targets are corrupted on blown-up rows. A GPU version needs multi-limb integer division in the loss, which is yet more hand-coded arithmetic.
- srt_sim.py contains `assert ... or True`, which always passes, so np.roll silently drops the top column. An overflow can then look like a broken congruence instead of an error. I saw this silent truncation in my replications, so harness checks built on this file can mislead.
- Without labels, the claim of 'learning from 1000 T=1 rows' measures nothing. Every row of any T, or random input, gives exact targets for P. The T=1 bottleneck analysis in the recipe therefore does not apply to the component that is actually learned.


## fixes

- Exit: use the candidates r+kN for k in {-2..2}, or do two correction rounds selected by sign. With the 6000-step P checkpoints this gave 768/768 on all 14 rungs for both seeds, assuming kappa and the gate are perfect. Another option is floor targets at the last step only, so that a +-1 error lands inside the covered range.
- Counter: break the permutation symmetry by giving D digit-value structure. One way is a circulant D(d) = d + k mod 10 with one shared learned shift; another is a scalar numeric counter. Then any single trained transition pins the whole table. Fixing Z and B is not enough: D alone still has 432 consistent tables. Also test training-time halting fits over many seeds before relying on it.
- Training: plan for at least ~6k effective steps of on-policy hindsight with in-loop noise and exit exploration, and check T=64 containment at checkpoints across several seeds. On H100 this is affordable.
- Remove torch.no_grad from the loop, and add a genuine differentiable path from the output loss to P. Otherwise rule 8 is broken outright.
- Be explicit that this is a designed solver with learned lookup tables. Within the rules' spirit, the only route is to make more of the datapath learned, and the design's own evidence says that does not bootstrap. As a research artifact, report it as a lower bound on how much must be designed, not as a compliant entry.
- Pick the OOD feature and margin hyperparameters on synthetic moduli at the OOD bit sizes, not on the certification profiles.


## hidden_assumptions

- That an exact column ALU plus SRT schedule, with tiny learned tables, counts as 'the learned model' under rule 7.
- That the loss may compute exact quotients of the model's internal register. That computation is label-free, so it is an algorithmic teacher.
- That one-hot counter tables learn the true decrement for digits that never appear in training T. Symmetry shows they cannot.
- That {r-N, r, r+N} covers the range of post-division remainders. It does not.
- That results from one CPU run per variant (256 rows, T=1 only) carry over across seeds and step counts, and to the multi-phase H100 recipe.
- That the on-policy and teacher-forced variants behave alike. They do not, and only the on-policy one is evaluator-compatible.
- That P's features are computed in true fp32: autocast is off and no TF32 matmul is used for the place-value readout.
- That a softmax population gate can choose the copy that generalises to unseen T, even though no training signal separates the copies.
- That jitter added for margin does not degrade the exact boundaries the exit needs.
- That the rows where counter or overflow errors break integrality are rare enough that dropping them does not starve the counter's training.


## algorithm_check

All scripts and logs are in /private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/panel/critique-training-dynamics/.

(1) Oracle SRT squarer (srt_sim.py): reproduced 768/768 on all 14 rungs, with max |r|/N 0.5 and max |col| 14. The congruent column datapath is correct.

(2) Noise robustness with the round-based oracle (robust.py): reproduced. Digit set [-9,9] overflows at 1% noise; [-19,19] tolerates +-1 errors at 30%. With the trunc semantics P is actually trained toward (robust_trunc.py): +-1 at 30% is fine (max |r|/N 2.0); +-2 at 10% overflows depth_t_64; +-3 at 0.1% overflows both profiles.

(3) Hindsight identity (hindsight_identity.py): the label-derived targets equal trunc(cur/(N*10^i)) in 2.4M of 2.4M cases, so they are label-free. There are 5 unreduced T=1 rows, as claimed.

(4) Training P with the design's evaluator-compatible on-policy+exploration script, unchanged, checked with its own T=64 test using an exact exit:
- 1500 steps: 0 of 8 seeds contained; all overflowed on both profiles.
- 2500 steps: 0 of 2 (train_P.py + test_exit.py); runaway by T=2-16.
- 6000 steps: 2 of 2 contained on both profiles (max |r|/N 2.0-2.95).
The teacher-forced variant, which is not compliant, worked 2 of 2 at 1500 steps (max |r|/N about 1.1).

(5) The design's learned exit (test_exit.py: 12-step P division, then select among {r-N, r, r+N}, kappa and gate assumed perfect), using the 6000-step checkpoints: it fails every one of the 14 rungs on both seeds, with 1-14 bad rows out of 768 and all failures at r < -N (post-division r/N as low as -1.07). With the 5-candidate fix, both checkpoints reach 768/768 on all 14 rungs.

(6) Counter (counter_ident_D.py, counter_train.py, counter_train2.py):
- Enumeration with Z and B fixed to true values: 432 D tables are consistent with training; 6 get T=32 right and 24 get T=64 right; the level of '3' is uniform over {3,5,7,9}.
- Soft halting-NLL training across 64 seeds, perfect exit assumed: 0 seeds fit T=8 and T=16 together.

(7) Compute: eval covers about 40.7k prompts (test 9,999; ood_t and ood_n_t 10,002 each at T=32; 14 x 768 depth rows). That is about 13-26k tiny sequential ops per batch, so seconds to minutes against the 1800 s budget. Training at 50-200 ms per step gives 3.6k-30k steps, enough for the roughly 6k steps that containment needed in my runs. Compute is not the problem.


## best_idea_to_keep

Keep the congruent redundant state. A designed carry-save column state with a wide quotient-digit set ([-19,19]) makes every in-loop decision tolerant of errors: a wrong quotient changes only the size of the representative, never the residue. Per-step exactness then no longer depends on exact learned decisions, the p^(T-1) decay disappears, and all exactness concentrates in one per-example exit. That exit should be a generous, exact sign-select over several candidates (r+kN, k=-2..2) rather than a precise learned boundary. Also worth keeping: the partial-credit criterion (modular reduction gives no partial credit to learned arithmetic upstream of it), and scale-free log features for the shift in N's leading digits on OOD moduli.



---

# Verdict: fatal  (p certify = 0.02)

## verdict

fatal


## probability_certifies_T64_both_profiles

0.02


## fatal_flaws

- The real system (arm A, q from the learned long division) gives no graded signal anywhere except right next to the solution. In hard execution on the 1000 T=1 rows, with k of the 1000 table fields set wrong at random: k=4 gives 45% exact and 55% digit agreement; k=12 gives 2.4% exact and 13% digit agreement; k>=30 gives 0.3% exact and 10.0-10.6% agreement on every digit, including the LSB, which is chance. Every label digit depends on q_0 (y_0 = P_0 - q_0*N_0 mod 10), and q_0 is the output of the last of 16 chained remainder rounds. So the region with any signal is a Hamming ball of about 15 fields in a table space of about 1.7 kbit, while identity init sits about 700 fields away. With q given (arm B) the landscape stays graded: LSB agreement is 57% at k=100 and 25% at k=300. The design's key claim that 'cells get graded partial credit' therefore holds only with an oracle quotient.
- Soft (mean-field) execution of the speculative long division is pathological even at the global optimum. I set SUB to the true table smoothed by eps, with PROD and ADD exact. Label NLL per digit is 6.69 at eps=1e-4, 4.81 at 1e-3, 3.55 at 1e-2 and 2.51 at 1e-1. That is worse than uniform (2.30), and the loss falls as the tables move away from the truth. Quotient-digit accuracy drops to 0% by the 3rd-5th round. The cause is that the soft remainder mixes in non-selected lanes whose wrapped digits (10^10 - jN, top digits 9) contaminate every later comparison; leakage grows roughly 50x per round. The consistent rule s_{j+1}-s_j fails the same way. The q-given verify path, by contrast, is benign: NLL 0.005, 0.05 and 0.46 at eps 1e-4, 1e-3 and 1e-2. Phase 1 as specified (soft execution) cannot produce a useful gradient for arm A.
- The circular dependency between the cells and q is confirmed even for a single cell under the most favourable conditions. Setup: only SUB is learned, PROD and ADD are frozen to the TRUE tables, Adam lr 0.1, full batch of the 1000 T=1 rows. With q given, SUB reaches 0/400 wrong fields and 100% exact in under 45 s. With q from the learned long division there is no progress in 461 steps soft (identity init), 361 steps with a straight-through hard lane select, or 226 steps with straight-through and uniform init. Loss stays at 2.02-2.05, quotient-digit accuracy is fixed at 2.3% (lane 9 always wins), and about 180/200 SUB digit fields stay wrong. With a wrong q, the verify subtraction's direct supervision drags the shared SUB into a hedge that is not subtraction, so the comparator never forms. Since 3 learned cells in arm A is strictly harder, the decisive experiment's arm A is already answered in the negative.


## serious_issues

- Rules spirit: the algorithm lives entirely in the wiring. Schoolbook squaring, a 16-round speculative long division, the multiply-back y = P - qN, the pointer parse of the digits and the int(T) loop are all designed. What is learned is about 1.7 kbit of single-digit arithmetic tables initialised to pass-through. A strict reading of rule 7 would likely disqualify it. On top of that come the comparator anchor (task facts x<N and y<N used as targets, with the loss running model cells on label digits), the PBT side channel from loss to optimizer, and hard-verified repair as program search, whose candidate evaluations would have to hide inside the forward pass to satisfy rule 14.
- Arm B, the upper bound, is unsolved by the design's own data: 733/1000 wrong fields after 1081 steps, 1.3% exact, LSB 79%. PBT, the curriculum, the anchors and repair are all untested. Repair only has signal within about 10 wrong fields (single-field corruption: median 84% exact; k=8 gives 11%; k=12 gives 2-4%), so it cannot pull a seed off a 700-field plateau.
- The LSB-first label curriculum conflicts with an MSB-first long division. The first label digit switched on (y_0) is a function of q_0, which is the deepest round.
- Under mean-field, the lane-select weights (1-s_j)s_{j+1} do not sum to 1 when the signs are soft. Measured leakage is 7e-5 after a single round from a 1e-6 perturbation, and it compounds round over round.
- The comparator anchor supervises only borrow-out semantics. It also admits comparators that are not subtraction (for example, a>b gives no borrow regardless of borrow-in). It never supervises SUB's digit outputs, so the remainder-digit circularity remains.
- Identity init sets borrow=0 everywhere. A final borrow then needs borrow generation plus propagation through about 10 slices of SUB(d,d,1) and SUB(0,0,1) entries, all initialised to 'no borrow', so the comparator gradient is a product of small factors.
- Phase 2 (8192 rows x 8 seeds x T<=16 with straight-through backward) has to store about 70 GB of soft activations; the 0.15-0.3 s/step estimate is roughly 5x optimistic. Blum-Blum-Shub labels make those gradients useless unless the machine is already exact.
- Data regime: with batch 8192 under the evaluator's shuffle, only about 160 T<=2 rows land in each batch, so the full-batch MPS probes do not transfer directly.
- A PBT population that starts from identical identity init plus 0.1 noise has little diversity.


## fixes

- Re-scope the decisive experiment. The first gate should be arm A with a single cell (SUB only, true PROD and ADD). It fails in my probes, which makes 3-cell arms A and B moot until the quotient source changes.
- Never mix candidate remainders softly. Snap the lane to its argmax in the forward pass every round, so each round is restored. This is necessary but not sufficient: my straight-through probe still failed.
- Give the comparator and SUB digits a supervision path that does not depend on an already-correct q. Options: (a) an LSB-first quotient source inside the step (Montgomery-style m_i = f(r_0, N_0), whose label dependence is triangular, built from the same 3 non-carry-save cells so coverage is kept), with top-down comparisons confined to a per-example setup and that setup's learnability tested on its own; (b) label-implied quotient targets used as teacher forcing in the loss, q = (P - y)*N^{-1} mod 10^k recovered LSB-first with the same learned cells. That is gray, because label digits flow into model cells inside the loss.
- Keep hard-verified single-entry repair, but only as a last-mile tool. Exact count is graded within about 10 wrong fields (k=4 gives 28-45% exact), and it should be triggered at about 10% exact rather than 50%.
- Set batch_size to 101,999 so every evaluator batch contains all 2k T<=2 rows (drop_last-safe, reuse each batch 8 times). Make Phase 2 forward-only hard verification on T=4/8/16, with no straight-through backward.
- Replace int(T) with the learned counter on the shared SUB cell (verified to extrapolate), and drop the comparator anchor or report it as a separate gray arm.


## hidden_assumptions

- That the label depending on q through a shallow path means shallow credit for the cells that produce q. It does not: q_0 is the last of 16 chained rounds, and every label digit, including the LSB, depends on it.
- That arm-B (oracle-q) learnability transfers to arm A. Measured: SUB-only reaches 100% with q given and makes zero progress with q learned.
- That mean-field soft execution approximates the hard machine near the optimum. True for the ripple scans and the verify path; catastrophically false for the speculative long division (NLL 6.7 at eps=1e-4).
- That k=1 near-miss partial credit (measured on Blakley) means a navigable landscape. In arm A the signal is gone by k of about 20-30 wrong fields.
- That identity init leaves the comparator trainable. It zeroes borrow generation and propagation everywhere.
- That PBT across 64 seeds with essentially identical init explores meaningfully different basins.
- That rule 14 permits per-seed exact counts to flow from the loss to the optimizer, and candidate-table trials to run inside the forward pass.
- That a hard-coded schoolbook/long-division datapath with learned digit tables satisfies rule 7.
- That the full-batch 1000-row probes represent training under evaluator-shuffled 8192-row batches holding about 160 T<=2 rows each.


## algorithm_check

I re-implemented the QX integer machine in vectorized PyTorch, following qexposed.py's semantics (WP=17, WR=10, WQ=16, WS=8; pairwise ADD tree; 10 lanes; select (0,1); y = P - qN). Scripts are in /private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/panel/critique-learned-wiring/: hardsim.py, corrupt.py (corrupt.log), graded.py, armA_sub.py (logs armA_sub_q0.log, armA_sub_st.log, armA_sub_st_ident0.log), soft_ld_stability.py, armB_smooth.py.

(1) With true tables the machine is exact on all 1000 T=1 rows and on all rows of depth_t_64, depth_ood_n_t_64 (including 110 rows with 8-digit N), depth_t_32 and depth_ood_n_t_32. Eval cost is trivial: about 1.2 s per 768-row T=64 split in CPU PyTorch.

(2) Coverage: all 500 entries are used by the T=1 rows. The minimum fraction of rows touching an entry is PROD 35.5%, ADD 8.3% and SUB 40.6%, and a row touches 261 distinct entries on average. The coverage claim holds.

(3) Single-field corruption (730 variants, tied as in the design): T=1 exact median 0.838, min 0.001, max 0.973. With k random wrong fields, exact rate is 0.75 at k=1, 0.29 at k=4, 0.105 at k=8 and 0.02-0.04 at k=12.

(4) Graded signal versus k (exact / digit agreement). Arm A: k=4 0.45/0.55; k=12 0.024/0.13; k=30, 100 and 300 about 0.00/0.10, which is chance on every digit. Arm B (q given): k=100 0.009 / LSB 0.57; k=300 LSB 0.25.

(5) Soft long division at the optimum (SUB smoothed): NLL per digit 6.69, 4.81, 3.55, 2.99 and 2.51 for eps 1e-4, 1e-3, 1e-2, 3e-2 and 1e-1, with quotient accuracy 0 by rounds 11-13. The q-given path gives 0.005, 0.05, 0.46, 1.13 and 2.02.

(6) SUB-only learning: 100% exact with q given in under 45 s; zero progress with q learned under soft, straight-through and uniform-init variants (60-120 s each on MPS).


## best_idea_to_keep

Coverage plus restoration. Factor the learned arithmetic into three tiny position-shared cells (PROD, ADD-with-carry, SUB-with-borrow) whose every entry the T=1 rows exercise heavily (all 500 entries, each touched by at least 83 of the 1000 rows). Then execute with argmax snapping after every micro-op at eval. Exact tables on T=1 then imply exact certification at T=64 and for 8-digit N; I verified this on all 1,536 T=64 rows, and eval costs seconds. Pair it with hard-verified single-entry repair, which has a real basin of about 10 wrong fields. The verify path y = P - qN is also worth keeping, but only once a quotient source exists that does not need a correct comparator to be supervised.



---

