# Guided search: what is known about letting a model order a program search

*Written 2026-10-29 for bar 3's gate study ([bar3/scale/gate/GATE.md](../bar3/scale/gate/GATE.md)). What the
program-synthesis literature knows about ordering an enumerative search with a learned model, mapped onto bar 3's
search. Each reference in §5 was checked against the paper's own text (arXiv, the ACM or the authors' PDF) on this
date; the numbers quoted are the papers'. One local check: the toy model of §3 (`bar3/scale/gate/toy.py`, seconds).*

## 1. In plain words

- **A search's cost is breadth to the power of depth.** DreamCoder's authors put it this way: a library of routines cuts
  depth (fewer calls per program) and raises breadth (more choices per call); a learned guide cuts breadth, "the
  number of bits of entropy consumed by each decision". Bar 3 measured both (SCALE.md §3): each instruction multiplies
  the states 9–10 times over the 35 primitives and 39–42 times over a library of 34 routines.
- **Two kinds of guide are known.**
  - *Conditioned on the problem only*: a model reads the examples once and prices the tokens for this problem.
    DeepCoder predicts which functions a program uses; DreamCoder's recognition model outputs a table of token
    probabilities per problem. Their baseline is the price bar 3's guided search uses (SCALE.md §4.4): how often each
    token appears in the corpus, the same for every problem.
  - *Conditioned on the values partial programs compute* (execution-guided): BUSTLE runs every intermediate expression,
    describes its values against the wanted outputs by a list of yes/no properties, and asks a classifier whether it
    is part of a solution. Unlikely values get a higher weight and so come later. Nothing is dropped, so the search
    stays complete. Probe does the same with no network: it raises the probability of the pieces of programs that
    already fit some examples.
- **What they found.** Problem-conditioned pricing gave DeepCoder 3–15 times faster depth-first search and up to
  hundreds of times faster enumeration over the corpus-frequency baseline. BUSTLE's execution guide solved more of its
  38 string tasks per candidate than both the plain search and DeepCoder-style selection, and the latter "does not
  lead to more tasks being solved overall" than no model. CrossBeam goes further and lets the model choose what to
  combine, giving up completeness.
- **What carries over to bar 3, and the catch.** Bar 3's search runs a register machine forward; a partial program is
  a machine state, five registers on every instance. A gate in BUSTLE's style reads that state against the problem's
  rows (is A below N? does a register equal x? equal the answer? share its parity? depend on N alone?) and delays the
  states it doubts. The catch is that partial progress may not show: in string editing a prefix of the answer looks
  like the answer, while here a loop does all the work in one step, and the state before it (say T = N) says little.
- **How strong a gate must be** (§3, a toy model): with branching 10, a gate that passes 20% of the wrong states and
  90% of the right ones (a likelihood ratio of 4.5) buys about 2 instructions per tenfold of states instead of 1; a
  ratio below 2 buys nothing, and a gate that delays right states often makes the search slower.
- **What the study tests.** Whether a gate trained only on the learner's own solved practice (its own programs and the
  values they compute; rule 14 allows no targets made by Python arithmetic) has that strength on problems it has not
  seen, at bar 3's depths. What it rediscovers: a fixed prior over tokens gives a constant factor, not depth
  (DeepCoder's baseline; bar 3's §4.4: x³'s program with a third of the states, no outcome changed).

## 2. The known methods

**DeepCoder** (Balog, Gaunt, Brockschmidt, Nowozin, Tarlow, ICLR 2017). A network maps the input/output examples to
"attributes" of the program, here the presence of each function of the DSL. Depth-first search tries functions in the
order of the predicted probabilities; "sort and add" searches with only the most probable functions, adding the next
when it fails. The baseline uses "a simple prior as function probabilities, computed from their global incidence in
the program corpus", which is bar 3's §4.4 price. Speedups to solve 20–60% of the tasks: 2.9–15.2× (DFS) and
31.5–62.2× (sort and add) on programs of length 3; on length 5, trained on length 4, 2.6–6.8× (DFS) and up to 907×
(sort and add, 20% of tasks). The guide sees the problem, never a partial program.

**Property signatures** (Odena, Sutton, ICLR 2020). A property is a yes/no function of an input and an output; over a
set of examples it is AllTrue, AllFalse or Mixed, and the list of these is the signature. A network fed signatures
predicted which functions a program uses; with it a distributed synthesizer solved on average 73 of 185 test programs
in 100,000 s, against 28 without, "twice as many programs in less than one-tenth of the time".

**BUSTLE** (Odena, Shi, Bieber, Singh, Sutton, Dai, ICLR 2021). Bottom-up enumeration by weight, where every new
intermediate value is classified by a small network on two property signatures: inputs against outputs, and the value
against outputs. The probability is binned into δ ∈ {0, …, 5} (bins at 0.1, 0.2, 0.3, 0.4, 0.6) and the value's weight
becomes w + 5 − δ. Trained on synthetic tasks made by its own searches on random inputs (positives: sub-expressions of
the chosen output; negatives: other values of the search). Complete: CrossBeam's authors note it "will eventually
explore the entire space". On 38 new benchmarks it solved 31 within 30 million candidates or 30 s, more than the plain
search, hand-written string heuristics and DeepCoder-style premise selection; with the heuristics added, more still.

**CrossBeam** (Shi, Dai, Ellis, Sutton, ICLR 2022). The model chooses which explored values to combine next, drawn by
beam search in training and by sampling distinct candidates in evaluation; it is "trained on-policy using data
extracted from its own bottom-up searches". It "trades off completeness for exploration efficiency": no guarantee of
reaching every program.

**DreamCoder** (Ellis et al., PLDI 2021). Wake: search each task "by enumerating programs in decreasing order of their
probability under the recognition model Q(ρ|x)". Sleep: compress solutions into a library, and train Q on replays
(the programs found while awake) and fantasies (programs drawn from the library, run to make tasks). Q is
task-conditioned but run once per task: it outputs a bigram table over the library, indexed by parent, child and
argument position. It does not see the values of partial programs. The unigram version it improves on (from EC²)
prices each token for the task independently of the others.

**Probe** (Barke, Peleg, Polikarpova, OOPSLA 2020). Bottom-up enumeration in order of probability under a grammar that
is learned during the search, for this one problem, from partial solutions: programs that already satisfy some of the
examples. Execution-guided, with no prior training and no dropped program.

## 3. What a delaying gate can buy: a toy model

Bar 3's search is uniform-cost by levels: level d holds every state whose cheapest program costs d (SCALE.md §1). A
gate in BUSTLE's style keeps that and adds a delay: each new state gets Δ ∈ {0, …, D} from the model, and its children
cost Δ more. Every program of plain cost c is still reached, at a cost of at most c + D·(c − 1): reordered, never
dropped.

Take a tree with branching b. Off the solution path the gate passes a state (Δ = 0) with probability q and delays it
by D otherwise; on the path it passes with probability r (the recall). States of cost at most C number about λ^C, with
λ the root of 1 = bq/λ + b(1 − q)/λ^(1+D), and an L-instruction solution costs about L + D(L − 1)(1 − r). So each
instruction costs b_eff = λ^(1 + D(1 − r)) states (`toy.py`, best D per row):

| b | q (wrong states passed) | r (right states passed) | r / q | best D | b_eff | instructions per tenfold of states |
|---|---|---|---|---|---|---|
| 10 | 0.5 | 0.7 | 1.4 | 1 | 9.95 | 1.00 (plain 1.00) |
| 10 | 0.5 | 0.9 | 1.8 | 1 | 6.99 | 1.18 |
| 10 | 0.3 | 0.8 | 2.7 | 2 | 5.90 | 1.30 |
| 10 | 0.2 | 0.9 | 4.5 | 5 | 3.19 | 1.98 |
| 10 | 0.1 | 0.9 | 9.0 | ≥ 12 | 1.84 | 3.77 |
| 40 | 0.2 | 0.9 | 4.5 | 2 | 12.95 | 0.90 (plain 0.62) |
| 40 | 0.1 | 0.9 | 9.0 | 3 | 6.90 | 1.19 |

So:
- the gate must tell right states from wrong ones by a likelihood ratio of about 3 or more, at every depth, before it
  buys a third of an instruction per tenfold; a weak gate buys nothing, and too large a delay with imperfect recall
  makes the search slower than plain (the cost of a missed right state compounds along the path);
- the gain is a change of slope, not a constant factor, only if that ratio holds as programs get longer. A gate
  trained on short solutions and tested on longer ones (DeepCoder's 4 → 5) tests exactly that.

## 4. What bar 3 adds, and what this study tests

- **Rediscovered:** a fixed prior over tokens (DeepCoder's baseline; DreamCoder's generative model; bar 3's §4.4)
  gives a constant factor. Bar 3's guided search found x³'s 4-token program with a third of the plain search's states
  and reached about 3 bits further per tenfold, where a token costs 4–9 bits; it changed no outcome in 40 runs.
- **Different from the literature:**
  - *the search space.* Not expressions built bottom-up but machine states run forward, merged when equal: a partial
    program is a state of five registers on every instance, and many programs share it;
  - *partial progress.* BUSTLE's string values show progress (a prefix of the output, the right length). A counting
    machine's loop computes in one step, and the states before it may show nothing; the gate can only use what the
    states show;
  - *training data.* No fantasies made by Python arithmetic (rule 14 as REVIEW.md reads it): only replays, the
    learner's own solved practice, which are short programs (1–6 instructions in a round from scratch). The gate must
    generalize to the longer programs it is meant to reach.
- **Tested:** whether generic property-signature tests on a state against the problem's rows (below N, equal to x,
  equal to the answer, the same parity, constant in x at each N) separate solution-path states from the others strongly
  enough (§3: a ratio of about 3 or more) to cut the effective branching factor on held-out problems, and whether that
  changes what the learner invents.

## 5. References

Checked on 2026-10-29 against each paper's text: arXiv full texts for DeepCoder, property signatures, BUSTLE (v3) and
CrossBeam; the PLDI paper as posted by an author (people.csail.mit.edu) for DreamCoder; the arXiv page and the ACM DOI
for Probe.

- Balog, Gaunt, Brockschmidt, Nowozin, Tarlow (2017). *DeepCoder: learning to write programs.* ICLR 2017.
  https://arxiv.org/abs/1611.01989
- Barke, Peleg, Polikarpova (2020). *Just-in-time learning for bottom-up enumerative synthesis.* OOPSLA 2020 (Proc. ACM
  Program. Lang. 4, article 227). https://doi.org/10.1145/3428295 ; https://arxiv.org/abs/2010.08663
- Ellis, Wong, Nye, Sablé-Meyer, Morales, Hewitt, Cary, Solar-Lezama, Tenenbaum (2021). *DreamCoder: bootstrapping
  inductive program synthesis with wake-sleep library learning.* PLDI 2021. https://doi.org/10.1145/3453483.3454080
- Odena, Shi, Bieber, Singh, Sutton, Dai (2021). *BUSTLE: bottom-up program synthesis through learning-guided
  exploration.* ICLR 2021. https://arxiv.org/abs/2007.14381
- Odena, Sutton (2020). *Learning to represent programs with property signatures.* ICLR 2020.
  https://arxiv.org/abs/2002.09030
- Shi, Dai, Ellis, Sutton (2022). *CrossBeam: learning to search in bottom-up program synthesis.* ICLR 2022.
  https://arxiv.org/abs/2203.10452
- Levin (1973), and bar 3's measured base: [LARGE_NUMBERS.md](LARGE_NUMBERS.md) §3.1, [SCALE.md](../bar3/scale/SCALE.md)
  §3–§4.4.
