# Search, networks and gradients: what a learner's form decides

*Written 2026-10-29. Asked: is a search over programs less learned than a neural network? The rules ask for
"end-to-end learning" (rule 8), and REVIEW.md §5 asks first whether the deep file's climb, a program search inside
training, counts. This note gives four standard facts, proves the one that compares search with gradients, and runs two
tiny local checks: [`checks/search_and_gradients.py`](checks/search_and_gradients.py) (about 18 s on a laptop), and
[`checks/step_gates.py`](checks/step_gates.py) (about 12 minutes), for the follow-up question of whether one network
with gates trained by backprop can stand in for the search. Every reference in §7 was looked up.*

## 1. In plain words

<!-- PLAIN -->
**Form can't be what makes something learned.** Every search built from if-else can be built as a ReLU network, with
the same answers on every input. Every ReLU network is an if-else program. So "is it a network?" can't separate
learned from hand-wired: compiling a search into weights makes it look neural without making anything more learned.
By the same logic, a network whose weights copy tables the author's code computes is no more learned than the tables.

**What is learned is what the data decides.** A learner maps training rows to a model. The model's learned part is
what changes when the rows change. The designed part is the learner itself: the candidates it can return, and the
order it prefers them in. Gradient descent is a learner of this kind, a local search over weights; so is a search over
programs.

**On the same candidates, a search and gradients keep the same ones.** Put trainable weights on a fixed list of
candidates and train them by gradient descent on the labels. The weights end up on exactly the candidates that fit
every row, which is the set a search's test keeps (§4.3, proved). The two differ only in ties (the search takes its
first, gradients split) and when nothing fits (the search says none; gradients settle on a blend).

**The tiny check agrees** (§5.1). The law reader is 14 laws of x and 1 under + and ×, mod N.
- The search compiled into a ReLU network of 6,041 units matched the search on all 6,200 tasks tried.
- Gradients put at least 0.9997 of their weight on the fitting laws in all 842 tasks where some law fits.
- On 14 tasks whose rows pin down one law each, all three pick right: the data decides 3.81 bits. A hard-wired law
  decides 0.
- A random ReLU network, rewritten as 98 lines of if-else, matched itself everywhere: one affine formula on each of
  its 130 regions.

**Backprop through one network doesn't get around the list.** Gates trained by backprop give the gradient path rule
8 asks for (§2). Over a list of whole programs, they keep what the search keeps, as above. Gates at every step fit in
memory however long the program, but they don't keep it. A second tiny check trained them by backprop on programs of
three instructions (§4.4, §5.2). They found a fitting program in 23% of runs at 8 steps (53% at the best settings
tried), 3% at 12, 1 run in 200 at 16 and none at 24 or 32. The search always finds one, at a cost that triples with
every step.

**What this settles, and what it doesn't.** It settles form: whether a choice is made by a search or by gradients, or
written as a network or as if-else, doesn't by itself change what is learned. It doesn't settle the candidates and what
computes them. Those are REVIEW §5's rule 7 and rule 14 questions. Nor does it bind a judge who reads "end-to-end
learning" by its letter.

## 2. The question

Rule 8 reads: "End-to-end learning only. Final logits must be produced entirely by the submitted model from its inputs
and learned PyTorch state, with all input-dependent computation inside the autograd graph and an unbroken gradient path
from the loss to the parameters responsible for the prediction." The deep file learns its round by searching programs
of its machine's instructions inside the training run, keeps the result in buffers, and returns zero logits in
training (REVIEW §5, questions 1 and 2). The rule's letter asks for a gradient path, not for training by gradients,
but a judge may read it as asking for a model trained by gradients through and through. The organizers named as
winner an entry whose digit cells are networks trained on targets its own code computes, with the procedure wired
by hand ([grading/GRADING.md](../grading/GRADING.md) §3). So the form of a part, network or search, is doing work in
how entries are judged, and it is worth knowing what form can and can't decide.

## 3. Definitions

- **A learner** L maps a task's training rows D to a model, m = L(D). The author designs L, and with it everything L
  can return: its candidate space.
- **What the data decides**, over a family of tasks f with rows D_f, is the map f ↦ L(D_f). A crude measure is
  log₂ |{L(D_f)}|, the bits the data selects: the number of different models the rows can lead to. A hard-wired model
  scores 0. A choice among K candidates scores at most log₂ K.
- **Gradient descent** is a learner. Its candidates are the weights the architecture allows; its order comes from the
  initialization and the optimizer's steps.
- **A search** is a learner. Its candidates are the programs it can enumerate; its order is its enumeration order. A
  search that returns the shortest program consistent with the rows is a PAC learner by Occam's razor (Blumer et al.
  1987; [LARGE_NUMBERS.md](LARGE_NUMBERS.md) §3.1).

## 4. Four facts

### 4.1 Every search built from comparisons is a ReLU network

On integers within known bounds, these gadgets are exact. They use only weighted sums and max(0, ·):

| computes | as |
|---|---|
| [z ≥ 1] | max(0, z) − max(0, z − 1) |
| v², for 0 ≤ v ≤ V | Σ over k < V of (2k + 1)·[v > k] |
| a·b, for a, b ≥ 0 | ((a + b)² − \|a − b\|²) / 4, with \|d\| = max(0, d) + max(0, −d) |
| v mod N, for 2 ≤ N ≤ M | v − Σ over q = 1..⌊V/2⌋ of max(0, N − M + M·[v ≥ qN]) |
| [a = b] | max(0, 1 − \|a − b\|) |
| all of c_1..c_R | max(0, c_1 + … + c_R − R + 1) |
| first fitting candidate i; none | max(0, c_i − Σ over j < i of c_j); max(0, 1 − Σ_j c_j) |

A search over a finite list of candidates, with bounded values, is a finite composition of these. So it is a ReLU
network whose weights the compiler fixes.
- **Precedents.** McCulloch and Pitts (1943) built logic from threshold units. Siegelmann and Sontag (1995) showed
  recurrent networks can simulate any Turing machine. Tracr (Lindner et al. 2023) compiles programs written in RASP
  (Weiss et al. 2021) into transformer weights.
- **What compiling preserves.** The compiled network computes from the rows what the search computed. So the data
  decides the same bits, and compiling moves no decision from the author to the data or back.

### 4.2 Every ReLU network is an if-else program

max(0, z) is "if z > 0 then z, else 0", so a ReLU network is a program of weighted sums and if-else. Each pattern of
signs fixes a region where the network is one affine formula. ReLU networks compute exactly the continuous
piecewise-linear functions (Arora et al. 2018), with a number of regions that can grow exponentially with depth
(Montúfar et al. 2014).

### 4.3 On fixed candidates, gradients keep what the search keeps

**Setup.**
- Candidates h_1..h_K have fixed outputs.
- Rows are r = 1..R, and a_ri = 1 if h_i gives row r's label, else 0.
- Trainable logits give weights p on the simplex by softmax.
- The model gives row r's label the probability (Ap)_r: a mixture of the candidates' answers, as in a softmax
  selector over a program list.
- The log-likelihood of the labels is ℓ(p) = Σ_r log (Ap)_r.

**Claim.** ℓ is concave. If some candidate fits every row, the weights that maximize ℓ are exactly those supported on
C = {i : a_ri = 1 for every r}.

**Proof.** (Ap)_r ≤ Σ_i p_i = 1, so ℓ ≤ 0. Then ℓ = 0 if and only if (Ap)_r = 1 for every r, which holds if and only
if no weight sits on a candidate that misses some row: p is supported on C. Concavity: the log of a linear function
is concave, and sums keep it. Smoothing for rows no candidate explains, q_r = (1 − ε)(Ap)_r + ε/N_r, changes nothing,
since log is increasing. ∎

So the search's test and maximum likelihood pick out the same set from the same rows. They differ in three ways:
- **Ties.** The search returns the first member of C in its order, a designed prior (here shortest first). Gradients
  from equal starting weights split.
- **When C is empty.** The search reports none. ℓ's maximizer is then a blend that explains some rows in part: the
  "soft blends" failure of relaxations such as DARTS (Liu et al. 2019).
- **Cost.**

**What is proved and what is only checked.** EM's multiplicative update for these weights raises ℓ at every step
(Dempster et al. 1977), and the geometry of this concave problem is Lindsay's (1983). Plain gradient descent on the
softmax logits isn't proved here, only checked (§5.1).

### 4.4 Gates at every step don't keep what the search keeps

**Setup.** A program is L steps, each one of I instructions, so there are I^L programs.
- **Whole-program gates** put one weight on each program, as in §4.3: I^L weights.
- **Step gates** put one softmax on each step and give a program the product of its steps' weights: I·L weights. Run
  on a register that holds a distribution over values, each step moving it by the mix of its instructions, the step
  gates give each row's label exactly the probability of §4.3's mixture under these product weights. Gaunt et al.
  (2016) call training them "forward marginals gradient descent". It is backprop through a soft program.

So step gates are whole-program gates held to product weights.

**What changes.** Product weights are a curved surface inside the simplex, not a convex set: half on the program
"1 then 1" and half on "2 then 2" is a mixture no product of steps gives. So ℓ in the step weights needn't be
concave, and it can have local maxima whose programs fit some rows and miss others. For L = 1 the surface is the whole
simplex, and §4.3 holds.

**Known.** Gaunt et al. built this relaxation as one back end of TerpreT, a language for program induction, and found
constraint solvers well ahead of it. On a ring of K parities they proved at least 2^((K−1)/2) local optima. Untuned
runs from random starts reached the global one in 100% of runs at K = 4, 53% at K = 8, 14% at K = 16 and none at
K = 32, 64 or 128. The best of 100 tuned settings reached it in 100% up to K = 16, 70% at 32, 80% at 64 and none at
128.

**Checked here** (§5.2): the same falloff on programs of three instructions mod N, given every row of their tables.
Step gates found a fitting program in every run up to 3 steps, in 23% at 8 (53% at the best settings tried) and in
3% at 12, then next to none. Failed runs end in blends whose loss has stopped moving.

## 5. The tiny checks

### 5.1 Search, network and gradients

**Setup.**
- **Candidates.** Laws f(x) mod N with f built from x and 1 by + and ×, at most 5 tokens. They are deduplicated by
  what they compute on every x < N ≤ 7, and ordered shortest first. That gives 14 laws: x, 1, 2x, x + 1, 2, x², 3x,
  2x + 1, x + 2, x² + x, 3, x² + 1, 2x², x³.
- **Tasks.** A task is 6 rows (x, N, y) with N from 2 to 7.
- **Arithmetic.** The candidates are evaluated with exact integer arithmetic: this check is about form, not about what
  computes the candidates.

| check | result |
|---|---|
| the network equals the search | 6,041 ReLU units, 16,073 weights, 10 ReLU layers, nothing but weighted sums and max(0, ·). Same pick on 6,200 of 6,200 tasks: 4,200 drawn from the laws, 1,000 with one label flipped, 1,000 random; 1,989 have no fitting law. Outputs exactly one-hot |
| gradients keep the fitting laws | 3,000 steps of descent from equal weights, on every 5th task (1,240). 806 with one fitting law: at least 0.9999 of the weight on it. 36 with several: at least 0.9997 on them, the search's pick holding a median 0.50 of it, an even split of a two-way tie. 398 with none: a blend of 3.0 laws at 5% or more each, median top weight 0.67 |
| what the data decides | one task per law whose rows pin it down (found for 14 of 14). The search, the network and gradients each make 14 different picks, all right: log₂ 14 = 3.81 bits. A hard-wired x² mod N makes 1 pick: 0 bits |
| the network equals an if-else program | a random 2 → 16 → 16 → 1 network with integer weights, rewritten as 98 lines of if-else with no max: the same output on all 3,721 grid points. 130 regions on the grid; on each, one affine formula read off the weights matches |

### 5.2 Gates at every step

[`checks/step_gates.py`](checks/step_gates.py), about 12 minutes on a laptop's CPU.

**Setup.**
- **Programs.** L instructions on one register y mod N, each one of INC (y + 1), DBL (2y) and CUBE (y³). For N in
  {5, 11, 17} all three permute the residues, so no program collapses to a constant.
- **Tasks.** 50 random programs a length. A task's rows are its program's whole table, all 33 pairs (x, N): the most
  a learner could be given.
- **Learners**, over the same 3^L programs:
  - the search, in a fixed order;
  - whole-program gates, from equal weights;
  - step gates, from 4 random starts a program, logits drawn from N(0, 1).

  Both kinds of gates train with Adam, 2,000 steps at learning rate 0.1. A step-gate run succeeds if the program of
  each step's top instruction fits every row.

| L | programs | search: tried, median | whole-program gates | step gates: runs | step gates: programs, any of 4 starts |
|---|---|---|---|---|---|
| 1 | 3 | 2 | 100% | 100% | 100% |
| 2 | 9 | 6 | 100% | 100% | 100% |
| 3 | 27 | 16 | 100% | 100% | 100% |
| 4 | 81 | 45 | 100% | 97% | 100% |
| 6 | 729 | 316 | 100% | 74% | 100% |
| 8 | 6,561 | 3,311 | 100% | 23% | 62% |
| 12 | 531,441 | 286,512 | not run | 3% | 8% |
| 16 | 4.3 × 10⁷ | not run | not run | 1 run in 200 | 2% |
| 24 | 2.8 × 10¹¹ | not run | not run | 0% | 0% |
| 32 | 1.9 × 10¹⁵ | not run | not run | 0% | 0% |

- **Stuck, not slow.** No failed run settled on one instruction at every step (top weight 0.99 or more). Their loss
  moved by a median 0.7–1.3 × 10⁻⁴ over the last 1,000 steps. Their programs fit a median 21–61% of rows, against 9%
  by chance.
- **Not the settings.** Swept at L = 8 and 16, over learning rates 0.03, 0.1, 0.3 and 1 and starting spreads 0.01
  (near an even mix) and 1. At L = 8: 41–53% of runs from near-even starts, 21–24% from spread ones. At L = 16: 0%
  at all eight settings.
- **Cost.** Step gates have 3L weights. Whole-program gates have 3^L, and the search tries a median half of the 3^L
  programs.

## 6. What it means for the entry

**Settled: the form.**
- Whether a choice is made by a search or by gradients, or written as a network or as if-else, doesn't by itself
  change what the data decides.
- A judge who accepts gradient training of selection weights is accepting the same learner as a search over the same
  candidates (§4.3).
- A search wrapped in a network is no more learned than before (§4.1).

**Not settled: the candidates, and what computes them.** Two entries with the same form can differ there:
- **Sahil Verma's file:** a softmax selector over 254,100 programs, each evaluated by the author's integer arithmetic.
- **The deep file:** its laws are evaluated by its learned routines on its learned tables, and its round is assembled
  by search from those routines (REVIEW §1).

**One network, with gates trained by backprop.** It would meet rule 8's letter (§2) where the deep file is gray
(REVIEW §5, questions 1 and 2). But it keeps the search's guarantee only in one form:
- Gates over whole programs keep what the search keeps, but only over a list made in advance: 254,100 programs in
  Sahil Verma's file, 3^L programs in §5.2.
- Gates at each step give the guarantee up from two steps on (§4.4). In §5.2 they held to about 4 steps and fell off
  after.
- The deep file's searches are short in calls: its H1 round is 2 calls of routines it found earlier (REVIEW §3's
  logs). But they run over its 30 instructions and that library, with loops, and on numbers with far too many values
  for a register that holds a distribution over them (blending digits instead isn't a mixture of numbers). §5.2 tests
  none of that.

So such a network changes the form that rule 8 reads. Whether gates trained by backprop could stand in for the climb's
searches at its sizes is untested; the evidence here is against it past a few steps.

**The questions left to a judge.** Whether a candidate space was built around the answer is REVIEW §5's rule 7
question; the same file, unchanged, learns all six tasks drawn at random by its md5. Whether reading the labels may
decide what to practise is rule 14's. This note doesn't rule on either, nor on a reading of rule 8 by its letter.

## 7. References

Checked on 2026-10-29 on Crossref (DOIs) and arXiv, with two venues from the NeurIPS proceedings and OpenReview. arXiv's
journal field for Arora et al. reads "ICLR 2028", a typo; OpenReview lists the paper at ICLR 2018.

- Arora, Basu, Mianjy, Mukherjee (2018). *Understanding deep neural networks with rectified linear units.* ICLR.
  https://arxiv.org/abs/1611.01491
- Blumer, Ehrenfeucht, Haussler, Warmuth (1987). *Occam's razor.* Inf. Process. Lett.
  https://doi.org/10.1016/0020-0190(87)90114-1
- Dempster, Laird, Rubin (1977). *Maximum likelihood from incomplete data via the EM algorithm.* J. R. Stat. Soc. B.
  https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- Gaunt, Brockschmidt, Singh, Kushman, Kohli, Taylor, Tarlow (2016). *TerpreT: a probabilistic programming language
  for program induction.* arXiv (§5.1–5.2 for the parity chain; Table 3 for the rates). https://arxiv.org/abs/1608.04428
- Lindner, Kramár, Farquhar, Rahtz, McGrath, Mikulik (2023). *Tracr: compiled transformers as a laboratory for
  interpretability.* NeurIPS. https://arxiv.org/abs/2301.05062
- Lindsay (1983). *The geometry of mixture likelihoods: a general theory.* Ann. Statist.
  https://doi.org/10.1214/aos/1176346059
- Liu, Simonyan, Yang (2019). *DARTS: differentiable architecture search.* ICLR. https://arxiv.org/abs/1806.09055
- McCulloch, Pitts (1943). *A logical calculus of the ideas immanent in nervous activity.* Bull. Math. Biophys.
  https://doi.org/10.1007/BF02478259
- Montúfar, Pascanu, Cho, Bengio (2014). *On the number of linear regions of deep neural networks.* NeurIPS.
  https://proceedings.neurips.cc/paper_files/paper/2014/hash/fa6f2a469cc4d61a92d96e74617c3d2a-Abstract.html
- Siegelmann, Sontag (1995). *On the computational power of neural nets.* J. Comput. Syst. Sci.
  https://doi.org/10.1006/jcss.1995.1013
- Weiss, Goldberg, Yahav (2021). *Thinking like transformers.* ICML. https://arxiv.org/abs/2106.06981
