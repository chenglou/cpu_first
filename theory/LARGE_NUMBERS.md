# Large numbers: what is already known, and what it means for H1

*Written 2026-10-25. What classical mathematics, learning theory and neural-network theory say about computing and
learning with very large numbers, mapped onto this project. Project facts come from [OVERVIEW.md](../OVERVIEW.md),
[REVIEW.md](../REVIEW.md), [INTEGRATION.md](../integration/INTEGRATION.md) and the bar-3 write-ups. Nothing was trained;
two small local Python checks were run for this note (§4.2, §5.2), in [`checks/`](checks/) (seconds each). Every
reference in §6 was looked up.*

## 1. In plain words

<!-- PLAIN -->
**The intuition, sharpened.** A number with k digits can be as large as 10^k. Counting up to it takes about 10^k
steps; walking over its digits takes about k. So the only programs that scale take a number one digit at a time,
each step multiplying the size covered by the base (by 2 in binary: the "doubling"). The "extrapolatable relation"
is that the step is the same at every position: then a step checked on short numbers is right on long ones, by
induction on the number of digits. Example: 1385 mod 7, reading digits from the top with r ← (10·r + d) mod 7, goes
0 → 1 → 6 → 5 → 6, and 1385 = 7 × 197 + 6. This is Horner's rule, the walk of long division. H1 has three sizes
that grow, each with its own doubling:
- **x and N** (up to 8 digits at test): the digit step. Stage 3 computes on digits, so it has this one.
- **the exponent 2^T** (up to 2^64): each squaring doubles it, x^(2^t) → x^(2^(t+1)). The round looped T times is
  this doubling; in log-log coordinates H1 is linear in T.
- **T itself**: x^(2^t) → x^(2^(2t)) in one step takes t more squarings, unless one knows the order of the group mod
  N, which takes factoring N. That is a time-lock puzzle's trapdoor, and no way around it is known.

**What is known.**
- Positional notation is an exact code whose length is the log of the size. Adding, comparing, multiplying by a fixed
  number and x mod N for a fixed N are finite automata over digits: one small table in a loop, right at any length.
  Multiplying two long numbers is not; it costs k² digit products by schoolbook, k·log k by the best known method.
- By multiplying alone, x^(2^T) takes at least T steps, and repeated squaring uses T. H1's function is Rivest,
  Shamir and Wagner's time-lock puzzle (1996), believed to need T sequential squarings without N's factors.
- Learning: short consistent programs generalize (Occam, MDL); the smallest automaton fitting data is NP-hard to find
  but easy with queries (Angluin's L*); learning automata from random examples can be as hard as factoring (Kearns
  and Valiant, using z, z², z⁴, … mod N, H1's own sequence). Their targets hide a secret; H1's are public.
- Neural: ReLU networks extrapolate linearly, so they extrapolate tasks linear in their inputs (Xu et al. 2021). Logs
  turn products into sums (NALU), but in floating point they round away the low digits a remainder needs. Modular
  addition is linear in clock (Fourier) coordinates and multiplication in discrete-log ones, learned for one small
  prime; for large unseen N those coordinates are the trapdoor.
- The paper recalled is most likely Odrzywołek's (arXiv, March 2026, not 2025): eml(x, y) = exp(x) − ln(y), with
  the constant 1, builds every function of a scientific calculator. It concerns real closed forms, not exact integer
  algorithms: a product is a 17-symbol tree, and fitting trees by gradient rarely succeeds past depth 4.

**So, exp/log as a reparametrization.** It already works here in three exact forms: digits (the log of the size),
repeated squaring (the log of the exponent) and, for a small fixed modulus such as the base inside a digit cell,
clock coordinates. It cannot work in the form that would matter most, coordinates in which x^(2^T) mod N is linear
for a large unseen N: they exist, but computing them is factoring and discrete log.

**Predictions.** Stage 3 already has the shape the theory asks for: digits, schoolbook multiply and long division
(finite digit cells in loops over places), and a round looped T times. For the digits study, "is the answer for
x·B + d a function of the answer for x, d and N?" asks whether the answer is its own automaton state. At N ≤ 7, in
unary, base 2 and base 10 alike (checked in Python): yes for x mod N, 2x + 1 and 2^x; no for x², x² ± 1, x² + 3,
10·x², 10·x² + 1, x³ and Montgomery's, which need x mod N as hidden state; no for Fibonacci, which needs a pair.
Unary cannot meet the step budget. Base 2 should come first: its Horner step needs at most one "subtract N if you
can", the link bar 3 already invents; base 10 needs up to nine.
<!-- /PLAIN -->

## 2. Classical theory

### 2.1 Positional notation is a code in log space

A number n in base B takes ⌊log_B n⌋ + 1 digits, n = Σ dᵢ·Bⁱ: exponential place weights, small digits, length the
log of the size, nothing lost. The other two ways to cope with size each lose what H1 needs. Floating point and
logarithms keep the leading digits and the exponent; products become sums of logs, but low digits are rounded away,
and x mod N depends on every digit. Residues (Chinese remaindering, Knuth §4.3.2) keep x mod m₁, …, x mod m_j for
fixed small moduli; products need no carries, but comparison and division, so reduction mod a new N, become hard.

### 2.2 Digit-serial arithmetic is finite-state; x mod N is Horner's automaton

Reading two numbers from the lowest place, a two-state machine (carry 0 or 1) adds them; subtraction keeps a borrow,
comparison "smaller so far", multiplying by a fixed c a carry below c: finite-state transducers, one small table at
every place, right at every length. The Myhill–Nerode theorem (Nerode 1958) says when such a machine exists: two
prefixes share a class when no continuation tells them apart, a function is finite-state exactly when the classes
are finitely many, and they are the smallest machine's states. For base k, Büchi's theorem (surveyed by Bruyère et
al. 1994): the sets definable from addition and "the largest power of k dividing x" are exactly those an automaton
recognizes in base k. x mod N for a fixed N is such a machine: from the top digit, r ← (r·B + d) mod N from r = 0,
N states, long division with the quotient thrown away. Its table has N·B entries: tiny for small N, hopeless for
H1's N (up to 2²⁴), where the step itself must be computed by a digit machine, one quotient digit at a time.

### 2.3 What is not finite-state: multiplication, and its cost

A middle digit of x·y depends on every pair of digits below it, so no fixed set of states can carry it. Rigorously,
an automaton checking z = x·y would make the first-order theory of addition and multiplication decidable, as Büchi's
theorem does for addition alone, and that theory is undecidable. For two k-digit numbers: schoolbook k²; Karatsuba
(1962) about k^1.585; Schönhage and Strassen (1971) k·log k·log log k; Harvey and van der Hoeven (2021) k·log k. At
H1's sizes (products of at most 16 digits) schoolbook is the right choice, and its loops run over places.

### 2.4 Automatic sequences (Cobham)

A sequence a(n) is k-automatic when an automaton reading n in base k outputs a(n) (Cobham 1972, "uniform tag
sequences"); equivalently its k-kernel, the subsequences n ↦ a(kᵉ·n + c) with 0 ≤ c < kᵉ, is finite (Allouche and
Shallit 2003).
Cobham's theorem (1969): a set recognizable in two bases that are not powers of one number, such as 2 and 10, is
ultimately periodic; the powers of 2 are recognizable only in bases that are powers of 2. For a fixed N every target
in bar 3's family is periodic in x (a function of x mod N, or of x modulo the period of 2^x or of Fibonacci mod N),
hence automatic in every base: the base changes the step's length, not whether it exists. What is not periodic,
like T or a number's length, can be easy in one base and impossible in another. Charton (2024) saw the base shape
learning: transformers computing GCDs in base B learn those built from the base's divisors and small primes.

### 2.5 Repeated squaring, square-and-multiply, addition chains

Each multiplication at most doubles the largest exponent reached, so x^e takes at least log₂ e multiplications.
Square-and-multiply takes at most ⌊log₂ e⌋ + (number of 1 bits) − 1, and the best chain is asymptotically log₂ e
(Brauer 1939; Knuth §4.6.3). For e = 2^T, T squarings meet the bound exactly: H1's exponent in binary is a 1 and T
zeros, and square-and-multiply on it is H1's loop. For 2^x mod N with x in binary it reads x's bits from the top,
t ← t²·2^d mod N, exactly the step the digits study would pose. Example, x = 13 = 1101₂ and N = 11: t goes
1 → 2 → 8 → 9 → 8, and 2¹³ = 8192 = 11 × 744 + 8.

### 2.6 Montgomery and Barrett reduction

Montgomery (1985) replaces division by N with division by R = B^k, a shift. For N coprime to B, each step reads only
the lowest digit t₀, picks q = −t₀·N⁻¹ mod B, adds q·N (clearing that digit) and shifts down a place; k steps give
t·R⁻¹ mod N up to one final subtraction. With B = 2 a step is "halve, adding N first when odd": bar 3's target
x²·2⁻¹ mod N is x² and one such step. Barrett (1986) precomputes ⌊B^(2k)/N⌋, estimates the quotient by multiplying
and corrects at most twice. H1's moduli are products of primes 3 more than a multiple of 4 (OVERVIEW.md §1), never
of 2 or 5, so N⁻¹ mod 10 exists and depends only on N's last digit: Montgomery's quotient digit in base 10 is a
one-place table of two last digits, the kind bar 3 invents (BRIDGE.md). Long division's quotient digit depends on the
leading digits of both numbers and is estimated, then corrected (Knuth §4.3.1, Algorithm D): the project's
quotient-digit puzzle (OVERVIEW.md §3) and its weak reduction check, fooled by repeated subtraction right only for
small quotients (INTEGRATION.md), sit there. With Montgomery, every quotient digit inside the loop over T is local;
the only reduction left outside it is R² mod N, once per modulus.

### 2.7 Time-lock puzzles and verifiable delay functions

Rivest, Shamir and Wagner (1996) lock a key behind b = a^(2^t) mod n for an RSA modulus n. The maker, knowing
φ(n) = (p − 1)(q − 1), computes e = 2^t mod φ(n) and then a^e quickly; computing φ(n) from n is as hard as
factoring, and repeated squaring "seems to be an 'intrinsically sequential' process". They call the puzzle an
application of the random-access property of the Blum–Blum–Shub generator (1986), x ← x² mod N for N = p·q with p
and q each 3 more than a multiple of 4: H1's moduli exactly, so H1's sequence over T is that generator's state.
Verifiable delay functions (Boneh et al. 2018) take many sequential steps to compute and few to check; Pietrzak
(2019) builds one on y = x^(2^T) mod N, "conjectured to require T sequential squarings", with a short proof. Katz,
Loss and Xu (2020) prove that in a strong algebraic group model any speed-up of it is as hard as factoring N.

### 2.8 Discrete logs and factoring: what they imply

For a prime p, write each nonzero x as g^i for a generator g; i is x's discrete log. Squaring doubles it,
i(x²) = 2·i(x) mod (p − 1), and T squarings multiply it by 2^T mod (p − 1). For N = p·q the same holds in two
coordinates, one per prime, with exponents reduced mod λ(N) = lcm(p − 1, q − 1). There H1 is linear and T costs
almost nothing, but the coordinates need discrete logs and λ(N), hence N's factors: the trapdoor. At H1's size even
the trapdoor buys little: λ(N) < N < 2²⁴, so the reduced exponent has at most 24 bits, at most 46 multiplications
against 64 squarings at T = 64, after factoring N. Either way the loop over T stays.

## 3. Learning theory

### 3.1 Occam's razor, MDL and Levin search

Blumer, Ehrenfeucht, Haussler and Warmuth (1987): hypotheses that fit the sample and are much shorter than it learn
in the PAC sense. Rissanen's MDL (1978) makes shortness the criterion; bar 3 compresses practised programs into
instructions and prefers what makes the target shortest. The guarantee covers the distribution sampled, not larger
sizes, and "short" depends on the instruction set (a constant 10 is long for a counting machine, BREADTH.md).
Levin's universal search (1973) runs each program p for a share 2^(−|p|) of the time, optimal up to a factor
exponential in the best program's length; bar 3 measured that base, 9–10 times more machine states per instruction,
about 40 over a carried library (SCALE.md). Libraries shorten targets; DreamCoder (Ellis et al. 2021), which solves,
compresses solutions into a library and trains a search policy, is bar 3's closest relative.

### 3.2 Identification in the limit (Gold)

Gold (1967): a learner identifies a class in the limit if on every presentation it changes its guess finitely often
and ends on a correct one. With every input labeled, the first consistent hypothesis of an enumeration does it for
any enumerable class of total functions; from positive examples alone not even the regular languages can be
identified. At a fixed size a look-alike can fit every row: bar 3's Montgomery target at odd N ≤ 7 has 16
instances, too few to pin down "halve mod N" (SUBGOAL.md). And the enumeration order decides which consistent
program wins, hence bar 3's checks at a bigger size (N ≤ 10 to accept, N ≤ 40 to call a program general).

### 3.3 Learning automata, and the cryptographic limit

- Angluin's L* (1987) learns the smallest automaton in polynomial time from membership queries and counterexamples.
- Without queries, finding the smallest automaton consistent with given data is NP-hard (Gold 1978). RPNI (Oncina
  and García 1992) merges states in polynomial time and is exact once the data hold a characteristic sample.
- Kearns and Valiant (1994): learning automata, formulas or small threshold circuits from random examples, with any
  polynomial-time hypothesis, would break RSA, factor Blum integers or detect quadratic residues. Their examples
  carry z, z², z⁴, …, z^(2^⌈log N⌉) mod N, so that decrypting becomes an iterated product; the concept is the key.

How our setting differs: there each target hides a secret and examples are random. In H1 the function is public and
the same for every N, N is in the prompt, and the honest algorithm is easy: T squarings. At bar 3's sizes the data
are complete (every pair x < N ≤ 7), and the learner poses its own practice and checks at a bigger size, the roles of
L*'s queries and counterexamples. Cryptography constrains not whether the algorithm can be learned but which: a
program that skipped squarings would be a factoring algorithm.

### 3.4 Length generalization: uniform programs against size-specific ones

A uniform program, one loop for every length, extrapolates by construction; a size-specific one, a circuit or table
per size, need not. Log-precision transformers of fixed depth are in uniform TC⁰ (Merrill and Sabharwal 2023). That
bound does not rule out one squaring mod N, since division and iterated multiplication are in uniform TC⁰ too
(Hesse, Allender and Barrington 2002). T squarings with T growing would need modular exponentiation in TC⁰, and no
efficient parallel algorithm for it is known at all (Fich and Tompa 1988): depth should grow with T, as stage 3's
loop does. And a finite-state step checked on every (state, digit) pair is right at every length by induction: a
proof, where a check at larger sizes is a test.

### 3.5 Why answers alone give gradients nothing

Shalev-Shwartz, Shamir and Shammah (2017) prove that for parities and other compositions of linear and periodic
functions the gradient is nearly the same whatever the target, for any architecture, so descent cannot find it, and
show that supervising sub-problems succeeds where end-to-end training stalls. A remainder is periodic: this is
OVERVIEW.md §2's "the remainder erases partial credit", and why the project teaches parts by laws and practice.

## 4. Neural theory, and exp/log as a reparametrization

### 4.1 How ReLU networks extrapolate

Xu et al. (ICLR 2021) prove that a ReLU MLP trained by gradient descent (in the neural tangent kernel regime) becomes
linear along every ray from the origin outside its data, so it extrapolates a target only when the target is linear
and the data cover all directions. Networks that extrapolate algorithms put the nonlinearity in the architecture or
the input features and leave the learned part linear: min-aggregation for Bellman–Ford, inverse-square features for
n-body physics. This is the exact form of the hunch that linearity works better in exp/log space: find coordinates
in which the task is linear. For H1 the question is which those are, and whether they can be computed.

### 4.2 Multiplying in log space: NALU and its successors

NALU (Trask et al. 2018) multiplies by exp(W·log(|x| + ε)), sums in log space, gated against an additive path. Madsen
and Johansen (2020) found the log path singular near zero, defined only for positive inputs, without a good
initialization, and the gate close to random; their NAU and NMU drop the log (and division). For exact arithmetic the
deeper problem is precision. A check for this note ([`checks/fib_and_float.py`](checks/fib_and_float.py), 200,000 random
x each): exp(2·ln x) in float64, rounded, misses x² for 17% of x in [2²³, 2²⁴), the top of H1's moduli, and for 97% of x
in [5·10⁷, 10⁸); float32 misses almost always, since it holds integers exactly only below 2²⁴ while x² reaches 2⁴⁸. A
remainder needs every low digit; a log drops them.

### 4.3 Clocks, characters and discrete logs

Networks trained on a + b mod p learn it by mapping inputs to cosines and sines of a few frequencies and combining
them with trigonometric identities (Nanda et al. 2023); Zhong et al. (2023) found a second algorithm ("pizza") beside
this "clock", and Chughtai et al. (2023) extended the picture to any finite group through representations. With
characters χ of Z_N, χ(a + b) = χ(a)·χ(b): addition is a rotation, linear in the complex exponential. Multiplication
mod p is addition of discrete logs, so the same works in discrete-log coordinates: Doshi et al. (2024) build networks
for it that way and conjecture that a modular polynomial is learnable when it has the form h(g₁(a) + g₂(b)); Nguyen
(2026) reports a grokked transformer for a·b mod 113 whose features are sparse in multiplicative characters. For H1
these coordinates fail twice: they change entirely with N (generator, group order, factors), and for large N
computing them is the discrete-log problem. OVERVIEW.md §3 saw it: generic networks learned addition as a rule but
the remainder only as a table, at chance on moduli never trained on.

### 4.4 Length generalization in transformers

Neural GPUs (Kaiser and Sutskever 2016) learned binary addition and multiplication up to 20 bits and made no errors
on much longer numbers in their tests. Zhou et al. (2024) conjecture that transformers length-generalize when a short
RASP program solves the task at every length. Abacus embeddings, which give each digit its place within its number,
take addition trained on 20 digits to 100 digits at up to 99% (McLeish et al. 2024).

### 4.5 The paper recalled: everything through exp

Odrzywołek, "All elementary functions from a single binary operator" (arXiv:2603.21852, March 2026, revised April
2026): eml(x, y) = exp(x) − ln(y), with the constant 1, rebuilds a scientific calculator's constants, arithmetic and
transcendental functions as binary trees of one node type, S → 1 | eml(S, S). It was found by exhaustive search and
computes through complex intermediate values on the principal branch of ln. Shortest trees found (its Table 4, in RPN
symbols): x − y 11, x·y 17, x + y 19, x² 17; an exp/log basis favours products, as NALU does. Trees fitted by Adam
recover exact formulas in all runs at depth 2, about a quarter at depths 3–4, under 1% at depth 5 and none of 448 at
depth 6. Usual activations are elementary, so every usual network is a special case. What it does not imply here:
- it concerns closed forms over the reals in floating point. Floor and remainder are not in its basis, and a
  wrap-around written with complex logs, such as (N/2π)·arg(e^(2πix/N)), needs as many significant digits as x has;
- it linearizes nothing: like NAND it is a universal basis, and expressivity is not what H1 lacks;
- as an instruction set it lengthens common operations (a product is 17 symbols), where bar 3 pays 9–10 times per
  instruction.

What carries over: one node type gives a regular search space, and exact formulas appear when weights snap to
discrete values, as the project's tables snap to exact digits.

## 5. What it means for this project

### 5.1 The map

| known idea | in this project |
|---|---|
| positional notation, a code whose length is the log of the size | stage 3's base-10 registers |
| digit-serial transducers: one table per place, in a loop | the digit cells (SBB, ADC, MUL, select) at a pointer |
| a transducer's step is one place of a law | BRIDGE.md's one-place slices: 11 of 18 compose back into their law |
| what one place cannot know: remainder, quotient, full product | the loops over places in stage 3's frame |
| Horner: append one digit | long division's step with BRG.X, and the digits study's question |
| schoolbook multiplication, k² cells | the round's multiply, a loop of accumulating rows |
| square-and-multiply, addition chains | the round looped T times; the binary step for 2^x mod N |
| automatic sequences: periodic in x means finite-state in any base | bar 3's targets at a fixed small N |
| time-lock puzzle, Blum–Blum–Shub generator: sequential in T | H1 itself, on Blum integers |
| MDL, Levin search, libraries (DreamCoder); L*'s queries | bar 3's merges and routines; self-posed practice |
| no gradient signal on periodic targets | "the remainder erases partial credit" (OVERVIEW.md §2) |

### 5.2 Predictions for the digits study

The study's test asks whether the answer is its own state (§2.2): equal answers must stay equal after any appended
digit. It fails when two residues share an answer but not their futures, and then the loop's state must be x mod N.
Checked in Python ([`checks/step_test.py`](checks/step_test.py); Fibonacci's pair state in
[`checks/fib_and_float.py`](checks/fib_and_float.py)) on every x < 2,000 (600 for Fibonacci, 400 for its pair), every digit d
and every N ≤ 7 (odd N for Montgomery's), in unary (x → x + 1), base 2 and base 10, with one verdict for all three:

| target | answer its own state? | state, and step x → x·B + d |
|---|---|---|
| x mod N | yes | r = x mod N; r ← (r·B + d) mod N, long division's step |
| 2x + 1, any a·x + c | yes | the answer t; t ← (B·(t − c) + a·d + c) mod N |
| 2^x, any c^x | yes | the answer t; t ← t^B·c^d mod N |
| x², x² ± 1, x² + 3, 10·x², 10·x² + 1, x³, Montgomery's | no | x mod N, then the small-N program once |
| Fibonacci | no | the pair (F_x, F_{x+1}): F_{2n} = F_n·(2F_{n+1} − F_n), F_{2n+1} = F_n² + F_{n+1}² |

Particular N break the "no" rows: x² at N = 3, 6 and 7 in base 10 (and 5 in base 2), x³ only at N = 7. Then:
- **Which come as one loop.** x mod N, 2x + 1 and 2^x should be posed and learned as a loop with the step as body.
  The polynomial targets need a piece first, x mod N, which SUBGOAL.md's table score should find (the target is a
  function of x mod N and N); the program found before at x < N then runs once at the end. Fibonacci needs a
  two-number step, the kind SCALE.md's source poses for x → x + 1, here for x → 2x + d.
- **Unary, base 2, base 10.** All pass the same tests, so they differ in cost. Unary needs about x steps, 10⁵–10⁸ at
  6–8 digits, and fails the budget. Base 2 has the cheapest step: 2r + d < 2N, so one "subtract N if you can"
  (PRACTICE.md's link) finishes it, and for 2^x a squaring and a doubling. Base 10 needs ×10, then up to nine
  subtractions: the full reduction, or a loop over the base as stage 3's machine has (counting up a constant 10 is
  what made 10·x² hard in BREADTH.md). For 2^x the base-10 step is a tenth power, four multiplications.
- **Extrapolation.** At small N every (state, digit) pair occurs among short numbers, so a step learned there is a
  complete table and the 6–8-digit check follows (§3.4), as with RPNI's characteristic sample: free while N is small.

### 5.3 Where exp/log reparametrization helps

- **Size, through digits.** Positional notation is the exact log reparametrization: it turns about 10^16 counting
  steps (BRIDGE.md's estimate at H1's size) into loops over 8–16 places. Stage 3 has it; the digits study gives it
  to bar 3.
- **The exponent, through log-log.** Repeated squaring makes the exponent's log the loop counter: stage 3's countdown.
- **A small fixed modulus, through characters.** Inside a digit cell the modulus is the base, 10, and addition mod 10
  is a rotation in clock coordinates: a Fourier digit code could speed up learning a cell, though not change what it
  computes, since every row of the table is seen.

### 5.4 Where it cannot help

- **Large unseen N, by cryptographic assumption.** The coordinates in which H1 is linear are discrete logs and λ(N),
  the time-lock trapdoor: factoring and discrete log for each new N, and little gain even at H1's size (§2.8).
- **Exactness, by counting digits.** A fixed-precision log or float code keeps a fixed number of significant digits;
  a product of two k-digit numbers needs 2k, and a remainder needs the lowest ones (§4.2).
- **Gradients, by theorem.** Where the target stays periodic, answers give gradients almost no signal (§3.5).
- **Expressivity is not the gap.** EML trees, NAND gates and the project's instruction set are all universal; what
  differs is how long the needed program is in each, and so how far search reaches.

### 5.5 Cheap tests

1. **The pose rule's truth table** (seconds, no search). Run the digits study's functional test alone on its rows for
   the 12 targets in unary, base 2 and base 10. Expected: §5.2's table; a "yes" for x², x³ or Fibonacci is a bug.
2. **The hidden state, found by the score** (seconds). Pose x mod N on long x and score it with SUBGOAL.md's table
   rule. Expected: first and alone for the seven polynomial targets; not for 2^x, whose state is the answer.
3. **Step length by base** (minutes, local search). Search the shortest programs for r ← (r·B + d) mod N and
   t ← t^B·2^d mod N at N ≤ 7, B = 2 and 10, over the round-0 library (link, square). Expected: base 2 within the
   usual 6-instruction reach; base 10 not, unless the machine loops over the base or has the reduction as a routine.
4. **Montgomery's quotient digit as a slice** (seconds). Add "q with t + q·N ≡ 0 mod B, then shift" to BRIDGE.md's
   slice book and run its one-place test and lift check. Expected: a function of one place (t₀ and N₀) that lifts to
   t·B^(−k) mod N, unlike long division's quotient: a reduction from stage 3's cells plus one new one-place table.
5. **The 2²⁴ test for any float or log cell** (seconds). Before training one, run it on H1's largest operands, x near
   2²⁴, products near 2⁴⁸. Expected (§4.2): float32 fails; float64 through exp∘ln misses about one product in six.

## 6. References

Checked on 2026-10-25 against publisher, arXiv, DOI or author pages. Two details rest on secondary records: Kaiser
and Sutskever's venue (DBLP lists ICLR 2016) and the DOI of Oncina and García's chapter (it resolves to the publisher).

- Allouche, Shallit (2003). *Automatic Sequences.* Cambridge University Press. https://doi.org/10.1017/CBO9780511546563
- Angluin (1987). *Learning regular sets from queries and counterexamples.* Inf. Comput.
  https://doi.org/10.1016/0890-5401(87)90052-6
- Barrett (1986). *Implementing the Rivest Shamir and Adleman public key encryption algorithm on a standard digital
  signal processor.* CRYPTO '86. https://doi.org/10.1007/3-540-47721-7_24
- Blum, Blum, Shub (1986). *A simple unpredictable pseudo-random number generator.* SIAM J. Comput.
  https://doi.org/10.1137/0215025
- Blumer, Ehrenfeucht, Haussler, Warmuth (1987). *Occam's razor.* Inf. Process. Lett.
  https://doi.org/10.1016/0020-0190(87)90114-1
- Boneh, Bonneau, Bünz, Fisch (2018). *Verifiable delay functions.* CRYPTO 2018. https://eprint.iacr.org/2018/601
- Brauer (1939). *On addition chains.* Bull. AMS. https://doi.org/10.1090/S0002-9904-1939-07068-7
- Bruyère, Hansel, Michaux, Villemaire (1994). *Logic and p-recognizable sets of integers.* Bull. Belg. Math. Soc.
  https://emis.univie.ac.at/journals/BBMS/Bulletin/bul942/BRU.PDF
- Charton (2024). *Learning the greatest common divisor: explaining transformer predictions.* ICLR 2024.
  https://arxiv.org/abs/2308.15594
- Chughtai, Chan, Nanda (2023). *A toy model of universality: reverse engineering how networks learn group
  operations.* ICML 2023. https://arxiv.org/abs/2302.03025
- Cobham (1969). *On the base-dependence of sets of numbers recognizable by finite automata.* Math. Syst. Theory.
  https://doi.org/10.1007/BF01746527
- Cobham (1972). *Uniform tag sequences.* Math. Syst. Theory. https://doi.org/10.1007/BF01706087
- Doshi, He, Das, Gromov (2024). *Grokking modular polynomials.* arXiv:2406.03495. https://arxiv.org/abs/2406.03495
- Ellis et al. (2021). *DreamCoder: bootstrapping inductive program synthesis with wake-sleep library learning.*
  PLDI 2021. https://doi.org/10.1145/3453483.3454080
- Fich, Tompa (1988). *The parallel complexity of exponentiating polynomials over finite fields.* JACM.
  https://doi.org/10.1145/44483.44496
- Gold (1967). *Language identification in the limit.* Inf. Control. https://doi.org/10.1016/S0019-9958(67)91165-5
- Gold (1978). *Complexity of automaton identification from given data.* Inf. Control.
  https://doi.org/10.1016/S0019-9958(78)90562-4
- Harvey, van der Hoeven (2021). *Integer multiplication in time O(n log n).* Ann. Math.
  https://doi.org/10.4007/annals.2021.193.2.4
- Hesse, Allender, Barrington (2002). *Uniform constant-depth threshold circuits for division and iterated
  multiplication.* JCSS. https://doi.org/10.1016/S0022-0000(02)00025-9
- Kaiser, Sutskever (2016). *Neural GPUs learn algorithms.* ICLR 2016. https://arxiv.org/abs/1511.08228
- Karatsuba, Ofman (1962). *Multiplication of many-digital numbers by automatic computers* (in Russian). Doklady
  Akad. Nauk SSSR. https://mathnet.ru/eng/dan26729
- Katz, Loss, Xu (2020). *On the security of time-lock puzzles and timed commitments.* TCC 2020.
  https://eprint.iacr.org/2020/730
- Kearns, Valiant (1994). *Cryptographic limitations on learning Boolean formulae and finite automata.* JACM.
  https://doi.org/10.1145/174644.174647
- Knuth (1997). *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*, 3rd ed. Addison-Wesley.
  https://www-cs-faculty.stanford.edu/~knuth/taocp.html
- Levin (1973). *Universal sequential search problems* (in Russian). Probl. Peredachi Inf.
  https://www.mathnet.ru/eng/ppi914
- Madsen, Johansen (2020). *Neural arithmetic units.* ICLR 2020. https://arxiv.org/abs/2001.05016
- McLeish et al. (2024). *Transformers can do arithmetic with the right embeddings.* NeurIPS 2024.
  https://arxiv.org/abs/2405.17399
- Merrill, Sabharwal (2023). *The parallelism tradeoff: limitations of log-precision transformers.* TACL.
  https://doi.org/10.1162/tacl_a_00562
- Montgomery (1985). *Modular multiplication without trial division.* Math. Comp.
  https://doi.org/10.1090/S0025-5718-1985-0777282-X
- Nanda, Chan, Lieberum, Smith, Steinhardt (2023). *Progress measures for grokking via mechanistic interpretability.*
  ICLR 2023. https://arxiv.org/abs/2301.05217
- Nerode (1958). *Linear automaton transformations.* Proc. AMS. https://doi.org/10.1090/S0002-9939-1958-0135681-9
- Nguyen (2026). *The discrete-log clock: how a transformer learns modular multiplication.* ICML 2026 Mechanistic
  Interpretability Workshop. https://arxiv.org/abs/2606.17399
- Odrzywołek (2026). *All elementary functions from a single binary operator.* arXiv:2603.21852.
  https://arxiv.org/abs/2603.21852
- Oncina, García (1992). *Inferring regular languages in polynomial update time.* Pattern Recognition and Image
  Analysis, World Scientific. https://doi.org/10.1142/9789812797902_0004
- Pietrzak (2019). *Simple verifiable delay functions.* ITCS 2019. https://eprint.iacr.org/2018/627
- Rissanen (1978). *Modeling by shortest data description.* Automatica. https://doi.org/10.1016/0005-1098(78)90005-5
- Rivest, Shamir, Wagner (1996). *Time-lock puzzles and timed-release crypto.* MIT/LCS/TR-684.
  https://people.csail.mit.edu/rivest/pubs/RSW96.pdf
- Schönhage, Strassen (1971). *Schnelle Multiplikation großer Zahlen.* Computing. https://doi.org/10.1007/BF02242355
- Shalev-Shwartz, Shamir, Shammah (2017). *Failures of gradient-based deep learning.* ICML 2017.
  https://arxiv.org/abs/1703.07950
- Trask, Hill, Reed, Rae, Dyer, Blunsom (2018). *Neural arithmetic logic units.* NeurIPS 2018.
  https://arxiv.org/abs/1808.00508
- Xu, Zhang, Li, Du, Kawarabayashi, Jegelka (2021). *How neural networks extrapolate: from feedforward to graph
  neural networks.* ICLR 2021. https://arxiv.org/abs/2009.11848
- Zhong, Liu, Tegmark, Andreas (2023). *The clock and the pizza: two stories in mechanistic explanation of neural
  networks.* NeurIPS 2023. https://arxiv.org/abs/2306.17844
- Zhou et al. (2024). *What algorithms can transformers learn? A study in length generalization.* ICLR 2024.
  https://arxiv.org/abs/2310.16028
