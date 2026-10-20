# Learning before the remainder

**Question.** From the digits of N and x, predict x² mod N. Does a network learn this better if it is also taught the
value before the remainder, x² (about twice as many digits as N)? And what if the network only has to write x², and a
fixed, exact remainder step turns it into the answer?

**TL;DR**
- **Teaching x² does not help the network take the remainder.**
  - Both models learn x² as an extra output (1.00 exact at 8 and 10 bits). Their answers are no better:
    - xf at 8 bits: 0.14 with x² vs 0.17 without (3 seeds each);
    - loop at 8 bits: 0.10 vs 0.12;
    - 10 bits: 0.04 either way.
  - Also teaching the quotient q = ⌊x²/N⌋ makes the answer worse.
  - With or without these targets, the network is right almost only where no reduction is needed (x² < N, so the answer
    is x² itself): 0.95–1.00 on those rows without the q target, less with it. Beyond one subtraction it is right on at
    most 0.09 of rows. That holds on trained and on held-out moduli.
  - A curriculum that fades the x² target out changes nothing.
- **The bottleneck is the remainder. At larger sizes the square is too.**
  - Given x² as an input, the remainder is no easier than the whole task: 0.19 vs 0.17 at 8 bits, 0.04 vs 0.04 at 10.
  - Inside the remainder:
    - The quotient is partly a rule: 0.63 exact, 0.56 on moduli never seen.
    - Multiplying it back and subtracting is not: 0.49 even when q is given.
    - One long-division step learns its quotient digit (0.99), but the step itself stays at 0.14–0.27.
  - x² alone is stored as a table at 8–12 bits (≥ 0.96 on seen x, 0.16 on held-out x). It is gone by 16 bits (0.02):
    only the lowest 3 and top 2–4 digits are right.
- **With a fixed exact remainder, x² is enough, and the answer alone can teach part of x² if the gradient is right.**
  - **x² supervised:** 1.00 exact, also on moduli never seen in training, at 8 and 10 bits. x² does not depend on N,
    so the stored squares carry over to every modulus.
  - **Answer only, through the exact remainder (8 bits):**
    - Straight-through gets 0.03, at 2k and 6k steps. It cannot reach the high digits of x².
    - Expected-cost lookahead and REINFORCE get 0.01. Within 30 steps they collapse to one constant internal value,
      where their gradient is exactly zero.
    - A **best-fix lookahead** works. It tries every single-digit change of the internal value through the exact
      remainder, and trains each digit toward the value that gives the most correct answer digits.
      - At 2k steps the internal value becomes exactly x² on every row with x² < 4N: 0.23 / 0.19 exact (2 seeds),
        the same on held-out moduli.
      - At 6k steps: **0.31–0.42 exact, 0.29–0.39 on held-out moduli** (5 seeds). A plain network trained as long
        memorizes its moduli (1.00) but gets 0.14 on held-out ones.
      - **It found the mirror identity x² ≡ (N − x)² (mod N) from answers alone.** It gets rows right by two routes,
        neither of which needs a reduction inside the network:
        - P = x² for small x;
        - P = (N − x)² for x near N.

        On held-out moduli these two routes are 97% of its correct answers (checked on 2 seeds).
      - Given either half of x², it learns the other half from answers alone: 0.998 and 0.999 exact.
  - **Answer only, 10 bits:** best-fix reaches only the x² < N rows (0.04 overall, like a plain network). Once x² has
    been learned, best-fix keeps it at 1.00 on answers alone; straight-through loses it (1.00 → 0.14).
- **The prediction "low digits learn, high ones don't" holds only for straight-through.**
  - Straight-through gives the high digits no gradient. Its low digits learn only when the high half is given (0.50–0.57
    per digit).
  - Best-fix learns x² row by row (small x first), not digit position by digit position. Given the low half, it learns
    the high half to 1.00.

## 0. Setup
- **Code:** `microlab/ext_preremainder.py` registers tasks `sqb`, `modsq`, `quot`, `subq`, `divstep`, `sqmodx` and the
  model `ht`.
  - Its `train()` copies `microlab.run.run` (checked: identical numbers for the same command). It adds:
    - accuracy of every auxiliary head;
    - held-out moduli;
    - exact match by quotient;
    - loss-weight and x-range schedules;
    - learning curves.
  - Rows: `microlab/results_preremainder.jsonl` (tags q1–q4); summary: `python -m microlab.ext_preremainder table q3`.
  - Logs and command lists: `runs/pre_remainder/`.
- **Models:** microlab's defaults.
  - `xf`: 3 layers, d = 128, 0.6M parameters.
  - `loop`: 1 shared block applied 6 times.
  - `ht`: the xf encoder writes the 2w digits of x² as an internal layer P. The answer is then computed by a fixed
    module: digits → integer → mod N → digits.
- **Training:** microlab's recipe: AdamW 5e-4, OneCycle, batch 256. Runs are 2,000 steps (512k fresh examples) unless
  noted.
- **Data:**
  - A b-bit modulus is any odd N in [2^(b−1), 2^b), and x is uniform in [0, N) (microlab's `sqmod`).
  - **Held-out moduli** (`--holdout 8`): every N with ((N − 1)/2) mod 8 = 3 never appears in training, and "ho" is
    exact match on 2,000 rows of those moduli. Tasks without N hold out x values by the same rule.
- **Table sizes and baselines:**
  - 8 bits: 12k (N, x) pairs, so 2,000 fresh steps see each pair about 40 times. 10 bits: 196k pairs, about 2.6 times
    each.
  - Rows with x² < N (q = 0): 7.6% at 8 bits, 3.7% at 10. Rows with q ≤ 3: 14.8% and 7.4%.
  - Chance on the 3-digit answer is about 0.01.
- **Seeds:** one seed per cell unless n is given. Repeated 2k-step cells agree within 0.05. The 6k-step best-fix runs
  spread 0.31–0.42 over 5 seeds.
- **Where it ran:** the first runs ran on the laptop (CPU 2 threads, loop on MPS). The rest ran on Modal CPU containers
  (4 threads), with the same code and seeds. The same seed on a different machine is not bit-identical.

## 1. Q1 — auxiliary targets
Exact on fresh rows of the trained moduli | exact on held-out moduli. In parentheses: exact of the auxiliary head (x²
or q) on fresh rows.

| model | aux | 6 bits | 8 bits | 10 bits | 8 bits, pool 1000 | 10 bits, pool 1000 |
|---|---|---|---|---|---|---|
| xf | none | 1.00 \| 0.22 | **0.17 \| 0.12** (n=3) | 0.04 \| 0.04 | 0.14 \| 0.05 (n=2) | 0.011 \| 0.009 |
| xf | x² | 1.00 \| 0.21 (1.00) | **0.14 \| 0.11** (1.00, n=3) | 0.04 \| 0.04 (1.00) | 0.16 \| 0.08 (0.94) | 0.021 \| 0.020 (0.43) |
| xf | q | 1.00 \| 0.19 (1.00) | 0.11 \| 0.11 (0.58, n=2) | 0.03 \| 0.03 (0.26) | 0.14 \| 0.06 | |
| xf | x², q | 1.00 \| 0.23 (1.00, 1.00) | 0.10 \| 0.11 (1.00, 0.52) | 0.03 \| 0.04 (0.99, 0.26) | 0.16 \| 0.07 | |
| loop | none | | 0.12 \| 0.11 (n=2) | 0.04 \| 0.04 | 0.14 \| 0.04 | |
| loop | x² | | 0.10 \| 0.11 (1.00, n=2) | 0.04 \| 0.04 (1.00) | 0.15 \| 0.08 (0.92) | |
| loop | q | | 0.09 \| 0.10 (0.50) | 0.02 \| 0.02 (0.21) | 0.14 \| 0.06 | |
| loop | x², q | | 0.09 \| 0.10 (1.00, 0.43) | 0.03 \| 0.03 (0.92, 0.19) | 0.15 \| 0.07 | |

**Which rows are right**: exact match by quotient q = ⌊x²/N⌋ on held-out moduli, listed as q = 0 / 1 / 2–3 / ≥4.

| run | q = 0 | q = 1 | q = 2–3 | q ≥ 4 |
|---|---|---|---|---|
| xf, 8 bits, none / x² (seed 2) | 0.98 / 0.96 | 0.33 / 0.26 | 0.06 / 0.06 | 0.05 / 0.02 |
| loop, 8 bits, none / x² (seed 2) | 0.98 / 0.95 | 0.19 / 0.21 | 0.00 / 0.00 | 0.03 / 0.03 |
| xf, 8 bits, q | 0.98 | 0.28 | 0.09 | 0.02 |
| xf, 10 bits, none / x² | 0.97 / 0.97 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| xf, 8 bits, pool 1000, none / x² | 0.42 / 0.79 | 0.00 / 0.07 | 0.00 / 0.00 | 0.02 / 0.01 |

- **x² is learned and not used.**
  - The x² head is ≥ 0.99 exact by step 500 at 8 bits and ≥ 0.97 by step 1,000 at 10 bits (0.92 at the end for loop
    with x² and q).
  - The answer stays at the plain network's level, or a little below it.
  - **Which digits.** Per-digit answer accuracy at 8 bits (xf), low digit first: 0.44 / 0.33 / 0.68 without x², 0.40 /
    0.28 / 0.65 with it. The majority baseline is 0.13 / 0.16 / 0.56; the top digit is bounded by N.
- **The quotient target hurts** in every model and size: xf at 8 bits 0.17 → 0.11, loop at 10 bits 0.04 → 0.02.
  - At 8 bits the q head learns q's top digits (0.97–1.00) but not its last digit (0.44–0.59).
  - That last digit is exactly the information that decides the remainder.
- **The only gain is on the rows where the answer equals x².** With 1,000 examples, x² supervision lifts held-out moduli
  from 0.05 to 0.08 (8 bits) and from 0.009 to 0.020 (10 bits). The gain comes from the q = 0 rows (0.42 → 0.79).
- **6 bits is pure memorization.**
  - The 768 pairs are learned whatever the aux target.
  - Held-out moduli score 0.19–0.23. That is mostly the q = 0 rows (17% of held-out rows, 0.88–1.00 right); the rest
    add a few percent.

## 2. Q2 — the square alone, or the remainder alone?
xf, 2,000 fresh steps. Each row gives the network more of the computation, from the end-to-end task down to single
steps.

| piece | input → target | 8 bits: exact \| held-out | 10 bits: exact \| held-out | by q, trained moduli, 8 bits: 0 / 1 / 2–3 / ≥4 |
|---|---|---|---|---|
| whole task | N, x → x² mod N (`sqmod`) | 0.17 \| 0.12 | 0.04 \| 0.04 | |
| remainder, x² given | N, x² → x² mod N (`modsq`) | 0.19 \| 0.10 | 0.04 \| 0.04 | 0.99 / 0.58 / 0.39 / 0.10 |
| ... with q as aux | | 0.10 \| 0.11 | 0.04 \| 0.04 | |
| quotient | N, x² → ⌊x²/N⌋ (`quot`) | **0.63 \| 0.56** | 0.34 \| 0.34 | 1.00 / 1.00 / 1.00 / 0.56 |
| multiply back, q given | N, x², q → x² − qN (`subq`) | 0.49 \| 0.20 | 0.07 \| 0.07 | 1.00 / 0.46 / 0.57 / 0.45 |
| one long-division step | N, R < N, d → (10R + d) mod N (`divstep`) | 0.27 \| 0.20 | 0.15 \| 0.16 | |
| ... chained over x²'s digits | 2w steps from R = 0 | 0.16 \| 0.14 | 0.05 \| 0.05 | |
| ... with its quotient digit as aux | | 0.14 \| 0.13 (q digit 0.99) | | |
| square | x → x² (`sqb`, x as in sqmod) | **1.00 \| 0.16** (held-out x) | 0.96 \| 0.16 at 12 bits | |

`loop` gives the same picture: modsq@8 0.12 | 0.10; sqb@8 1.00 | 0.23.

**The square across sizes** (`sqb`, xf; loop is within 0.07):

| bits | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|
| exact (seen x) | 1.00 | 0.96 | 0.02 | 0.00 | 0.00 |
| exact (held-out x) | 0.16 | 0.16 | 0.01 | 0.00 | 0.00 |

- At 24 bits the per-digit accuracy is 1.00 1.00 1.00 0.30 0.11 0.10 … 0.11 0.23 0.66 0.99 1.00 1.00 1.00. The edges
  are rules, and the middle digits are at chance.
- microlab's `sq` task samples a uniformly random digit length, so short numbers are common. There xf gets 1.00 / 0.79 /
  0.62 at 3/4/5 digits, and 0.63 / 0.67 / 0.57 on held-out x.

- **The remainder is the wall.**
  - Handing the network x² buys almost nothing: 0.19 vs 0.17 at 8 bits, and nothing at 10 bits.
  - What it learns is again the no-reduction rows.
- **Inside the remainder, the division is the friendlier half.**
  - The quotient is monotone in x² and roughly x²/N, so most of it is learned as a rule. It carries over to new moduli
    (0.56).
  - Multiplying q back by N and subtracting is a table. It reaches 0.49 on trained moduli and 0.20 on new ones, and
    falls to 0.07 at 10 bits.
  - One long-division step shows the same split. Its quotient digit is learned (0.99), but its remainder is right only
    when nothing is subtracted: 0.97–1.00 on those steps, 0.51 at most on the rest.
  - Chained over x²'s digits, the step's errors compound: 0.16 at 8 bits down to 0.004 at 16.
- **The square stops being learnable two sizes later than the remainder.**
  - x² is learned up to 12 bits, but as stored answers: unseen x score 0.16.
  - Past 12 bits it fails like multiplication does in the ladder (`LADDER.md`): right edge digits, middle digits at
    chance.
  - At the competition's 24 bits, these baselines cannot produce the value before the remainder either.

## 3. Q3 — half-transparent: the network writes x², an exact module takes the remainder
The model `ht`: the xf encoder writes 2w digits P. The answer is computed as (P mod N) by exact integer arithmetic. How
the answer's loss reaches P (`--est`):
- **st (straight-through).** The quotient is frozen. Answer digit j is treated as P's digit j shifted by the module's own
  offset, so its argmax is the exact answer. Cross-entropy on the answer then reaches P's low w digits only, because
  moving a high digit changes the quotient.
- **local (expected-cost lookahead).** Every single-digit change of P (2w × 10 of them) is run through the module. The
  loss is the expected number of wrong answer digits under each digit's softmax.
- **rf (REINFORCE).** 4 samples of P per row with a leave-one-out baseline. The reward is minus the number of wrong
  answer digits.
- **best (best-fix lookahead).** The same single-digit changes as local. Each digit is trained by cross-entropy toward
  the value(s) with the fewest wrong answer digits. Digits whose value does not change the answer get no loss.

Columns:
- **P = x²:** exact match of P with x².
- **P digits:** P's per-digit accuracy against x², low digit first. The majority baseline is 0.22 0.22 0.14 0.24 0.53
  1.00.
- **by q (ho):** exact match by quotient on held-out moduli, listed as q = 0 / 1 / 2–3 / ≥4.

8 bits, 2,000 steps unless noted:

| training signal | exact \| ho | P = x² | P digits lo→hi | by q (ho) |
|---|---|---|---|---|
| x² supervised only | **1.00 \| 1.00** | 1.00 | 1.00 … 1.00 | 1 / 1 / 1 / 1 |
| x² supervised + answer (st) | 1.00 \| 1.00 | 1.00 | 1.00 … 1.00 | 1 / 1 / 1 / 1 |
| answer only, st | 0.03 \| 0.03 | 0.02 | 0.27 0.10 0.11 0.13 0.28 0.27 | |
| answer only, st, 6k steps | 0.03 \| 0.03 | 0.02 | | 0.22 / 0 / 0 / 0.01 |
| answer only, local | 0.01 \| 0.01 | 0.01 | 0.11 0.23 0.15 0.24 0.53 1.00 (one constant P) | |
| answer only, rf | 0.01 \| 0.01 | 0.01 | same constant P as local | 0.05 / 0 / 0 / 0 |
| answer only, **best** | **0.23 \| 0.23** (seed 1: 0.19 \| 0.20) | 0.17 | 0.28 0.26 0.25 0.25 0.46 0.89 | **1 / 1 / 1** / 0.10 |
| answer only, **best, 6k steps**, 5 seeds | **0.31–0.42 \| 0.29–0.39** (mean 0.35 \| 0.33) | 0.13–0.21 | seed 0: 0.33 0.34 0.30 0.29 0.43 0.73 | **1** / 0.82–1 / 0.34–1 / **0.20–0.28** |
| answer only, best, pool 1000 | 0.19 \| 0.11 (pool 0.89) | 0.11 | | 0.88 / 0.54 / 0.26 / 0.01 |
| high half of x² given, st for the rest | 0.56 \| 0.54 | 0.50 | **0.55 0.57 0.50** 1.00 1.00 1.00 | 1.00 / 0.79 / 0.00 / 0.50 |
| high half of x² given, best for the rest | **0.998 \| 0.997** | 0.998 | 1.00 … 1.00 | 1 / 1 / 1 / 1 |
| low half of x² given, best for the rest | **0.999 \| 0.999** | 0.999 | 1.00 … 1.00 | 1 / 1 / 1 / 1 |
| *plain xf* | 0.17 \| 0.12 | | | 0.98 / 0.33 / 0.06 / 0.05 |
| *plain xf, 6k steps* | 1.00 \| 0.14 | | | 0.98 / 0.42 / 0.00 / 0.05 |
| *plain xf, pool 1000* | 0.14 \| 0.05 | | | 0.42 / 0.00 / 0.00 / 0.02 |

At other sizes:

| | exact \| ho | P = x² | by q (ho) |
|---|---|---|---|
| 6 bits, answer only, best | 0.93 \| 0.02 | **0.00**: P is a valid representative but never x² | 0.00 / 0.00 / 0.08 / 0.02 |
| 10 bits, x² supervised only | 1.00 \| 1.00 | 1.00 | 1 / 1 / 1 / 1 |
| 10 bits, answer only, st | 0.02 \| 0.02 | 0.02 | 0.37 / 0.03 / 0 / 0 |
| 10 bits, answer only, best | 0.002 \| 0.003 | 0.00 | 0 / 0 / 0.02 / 0 |
| 10 bits, answer only, best, 6k steps | 0.04 \| 0.05 | 0.04 | 0.97 / 0.30 / 0 / 0 |

**Basin test.** x² is supervised for the first 1,000 steps (it reaches 1.00), then only the answer trains the network
for 1,000 more:

| | st | local | best |
|---|---|---|---|
| 8 bits, final exact | 1.00 | 1.00 | 1.00 |
| 10 bits, final exact | **0.14**: 1.00 at the switch, 0.23 after 250 steps; P's digits fall to 0.37–0.48 | | **1.00** |

- **A designed exact remainder makes x² sufficient.**
  - The x² table is learned in about 500 steps.
  - The answer is then exact, and it carries over to every modulus: x² does not involve N.
- **The answer alone does not teach x² through the remainder with ordinary estimators.**
  - **Straight-through** reaches only the low half of P. The high half gets no gradient and drifts: the top digit, always
    0 in x², is right 27% of the time. A wrong high half puts P in the wrong quotient class, so the low half's targets
    are scrambled too. 3× the steps change nothing.
  - **Expected-cost lookahead and REINFORCE** mode-collapse.
    - In a probe from the same initialization, every one of 256 rows had the same P by step 30, and it stayed there.
    - With a one-hot softmax the lookahead gradient p·(cost − E cost) is exactly zero, and identical REINFORCE samples
      have zero advantage.
    - Both runs end at the same constant P < N, which makes the module a no-op.
- **Best-fix lookahead teaches x² from the answer alone, near a solution.**
  - It turns the exact module into a local verifier: each digit learns which of its 10 values makes the answer right,
    given the other digits.
  - From random initialization at 8 bits, it first learns exactly x² for every x up to about 2√N (q ≤ 3), by step 500.
    That carries over to held-out moduli (0.23 vs 0.12 for the plain network), and with 1,000 examples (0.11 vs 0.05).
  - **With more steps it finds a second route: the mirror identity.**
    - Large-q rows go from 0.10 to 0.20–0.28 on held-out moduli. For x near N it writes P = (N − x)². That is valid
      because x² ≡ (N − x)² (mod N), and it needs no reduction either, since N − x is small.
    - The histogram of k = (P − x²)/N gave it away. Apart from k = 0, the common k values were all odd (−115, −149,
      −171, −191, …), as k = N − 2x must be for odd N.
    - Checked directly on held-out moduli (6k steps). P = x² and P = (N − x)² together are 97% of the correct answers;
      other valid P are 1%:

      | seed | exact (held-out moduli) | P = x² | P = (N − x)² | other valid P |
      |---|---|---|---|---|
      | 3 | 0.289 | 0.171 | 0.110 | 0.009 |
      | 4 | 0.363 | 0.224 | 0.129 | 0.011 |
    - So from answers alone, through an exact remainder, the network learned to square, and to square the mirror image.
      It learned essentially no reduction. Almost everything it learned is N-independent arithmetic, which is why it
      transfers to moduli it never saw.
  - At 10 bits P has 8 digits. In 6k steps it gets only as far as the x² < N rows.
  - Given either half of x², it completes the other half exactly.
  - It holds a learned x² at 10 bits, where straight-through loses it.
- **The answer does not single out x².** Any P ≡ x² (mod N) gives the right answer.
  - At 6 bits, best-fix fits 93% by memorizing other representatives per (N, x), never x². It gets 0.02 on new moduli.
  - At 8 bits, x² is the easier representative for small x, and that is what it learns first.

## 4. Q4 — small follow-ups
- **Curriculum, x² first then fade it out** (xf, 8 bits):

  | x² weight schedule | exact | held-out | x² head at end |
  |---|---|---|---|
  | constant (n=3) | 0.135 | 0.108 | 1.00 |
  | linear decay to 0 by 60% of training | 0.143 | 0.115 | 0.65 |
  | off at 50% | 0.139 | 0.104 | 0.55 |
  | none (n=3) | 0.170 | 0.122 | |

  The x² head is 1.00 until the target fades, then decays. The answer never moves, so there is nothing to carry over.
- **Size curriculum for best-fix** (`sqmodx`): x is drawn from [0, f·N), with f growing from 0.1 to 1 over the first
  60% of training. It does not help:

  | | 2k steps | 6k steps |
  |---|---|---|
  | with the curriculum | 0.21 \| 0.20 | 0.31 \| 0.25 |
  | without | 0.23 \| 0.23 | 0.42 \| 0.39 |

  It does not help the others either: plain xf with the same curriculum (6k steps) gets 1.00 \| 0.11 (without:
  1.00 \| 0.14). Best-fix at 10 bits, x from 0.05·N up (6k steps), gets 0.04 \| 0.04 (without: 0.04 \| 0.05).
- **Basin and half-given tests:** see §3. With best-fix, the answer keeps x² exact once x² is known, and fills in a
  missing half. Starting from nothing, it reaches only rows near a solution.
- **Long division as a learned cell** (`divstep`): see §2. One step is learned only where it subtracts nothing. Its
  quotient digit is learnable (0.99), but the multiply-and-subtract is not. Chained, the step's errors compound.

## 5. Takeaways
1. **Supervising the pre-remainder value is the wrong place to add supervision.** The network can produce x² (up to 12
   bits), but going from x² to x² mod N is as hard as the whole task. The obstacle is the exact multiply-back and
   subtract, not the square. Supervising the quotient does not fix it: q's last digit is the hard bit, and the q target
   costs accuracy.
2. **Design the remainder and the picture flips.** With an exact remainder module, the network only has to learn the
   N-independent part. Taught directly, that part is perfect, and it transfers to unseen moduli. The remaining problem
   is learning x² itself at 24 bits, which these baselines cannot do (§2).
3. **Answers can teach through an exact module only by local speculation.**
   - Best-fix lookahead runs 2w × 10 candidate edits through the exact module and keeps the winners. It is REPORT.md's
     "labels select among candidates", applied one digit at a time inside the training loop.
   - Of the estimators tried here, it is the only one that teaches anything behind the remainder from answers alone.
     What it teaches transfers to unseen moduli.
   - What it learns is the arithmetic that avoids reduction: x² for small x, and (N − x)² for x near N, which it found
     by itself. It learns essentially no reduction: other valid representatives are 1% of rows.
   - Like every graded signal found so far, it works only within one edit of a solution: small x, its mirror, a
     half-known x², or a learned x² that must be held.
   - It does not discover x² at 10 bits from scratch.
4. **Which estimator matters more than which digits.**
   - Straight-through cannot see the high digits.
   - Expectation-style estimators (expected cost, REINFORCE) collapse to a constant with exactly zero gradient.
   - A log-likelihood target on the best single-digit fix keeps learning.

## 6. Plain-language summary
We asked whether showing a network the square before taking the remainder helps it learn "square, then take the
remainder." It does not. The network happily learns the square as a side answer, but it never uses that to get the
remainder right. It still gets right only the easy cases, where the square is already smaller than N and there is no
remainder to take. The remainder is the hard part. Even with the square handed over, the network can guess roughly how
many times N fits (the quotient), but it cannot do the exact "multiply back and subtract".

If we instead build the remainder step in as fixed, exact arithmetic, the network only has to produce the square. Taught
the square directly, it is then perfect, even for moduli it never saw. Taught only the final answers, it mostly learns
nothing, because a wrong square produces an unrelated remainder and gives no hint of how to fix it. One trick works: for
each digit of its square, the network tries all ten values through the exact remainder and learns the one that makes
the answer right. With that, final answers alone taught it to square small numbers and to finish a half-known square.
It even discovered on its own that x near N can be handled by squaring N − x instead, which gives the same remainder.
What it learned works on moduli it never saw. But it only works when the network is already within one digit of right.
So it learns the cases that avoid a real remainder, not the remainder itself, and on bigger numbers it gets no further
than the cases without a remainder.

## 7. Reproduce
From `cpu_first/`, capping threads on the shared laptop:

```bash
python -m microlab.ext_preremainder sqmod --size 8 --model xf --aux pre --holdout 8 --steps 2000 --eval_every 500 --threads 2
```

```bash
python -m microlab.ext_preremainder sqmod --size 8 --model ht --est best --holdout 8 --steps 6000 --eval_every 1000 --threads 2
```

```bash
python -m microlab.ext_preremainder table q3
```

Every command used here is in `runs/pre_remainder/*.txt` (`<log name> <args>` per line). `sweep.sh <list> <parallel>`
runs a list locally. `modal run microlab/ext_preremainder_modal.py --commands <list>` runs one Modal CPU container per
line (about 10 minutes for 40 two-thousand-step cells).
