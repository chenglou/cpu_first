# Learning x^(2^T) mod N from examples

One training run, on one H100 in under an hour, that teaches itself x^(2^T) mod N (task H1 of the One Layer Deeper
competition) from prompt/answer pairs. It reads the formula from the answers, sets its own practice, and searches for a
program of its digit machine's instructions that computes it, with no hand-written practice book.

**The entry** is [`integration/legal_deep/submission_stage3_deep.py`](integration/legal_deep/submission_stage3_deep.py)
(md5 `56acca7358c74c65c72b5f33e3b9aafa`). On the contest's unmodified evaluator it certifies H1 on 8 of 8 H100 runs,
exact up to T = 64 on seen and unseen moduli, training in 502–741 s of the 3,600. The same file, unchanged, learns all
22 other formulas it was run on, including 6 drawn at random with a seed taken from the file's own md5 (6 of 6). Its
limits: it was run only on formulas of the form a·y^k + b·y + c mod N (k 2 or 3, small coefficients); people still
designed parts of it, which REVIEW.md lists; and its margins of budget and time are thin.

**Start here.** [REVIEW.md](REVIEW.md) is for a judge: what is claimed, how to check it, what is still designed, and the
rules to rule on. [OVERVIEW.md](OVERVIEW.md) tells the whole story and has the map of every folder.
[grading/GRADING.md](grading/GRADING.md) grades the entry against the contest's archive of every submission. It ties
the top three on score. Our runner reproduces the hosted evaluator's stored results. And in a blind review next to the
top three, it's the only one not judged to hard-code the target.

**How to check.** This repo shares its name with the competition's, and the scripts look for the competition's clone at
`../one-layer-deeper`, so clone this one under another name. The record checker needs only Python's standard library;
REVIEW §4.1 has the commands to rerun the entry itself.

```bash
git clone https://github.com/chenglou/one-layer-deeper cpu_first
git clone https://github.com/tilde-research/one-layer-deeper
git -C one-layer-deeper checkout 87e5c26
cd cpu_first && python3 tools/verify_records.py   # 50 of 50 records check out
```

**Layout.**
- [`integration/`](integration/): the H1 model from its first stage to the entry, with its builders, checks and records
- [`bar3/`](bar3/): the self-practising learner on a bare counting machine; the entry's law reader comes from it
- [`tools/`](tools/): the shared Modal runner and `verify_records.py`
- [`grading/`](grading/): the entry graded against the contest's archive, with five past submissions rerun
- [`theory/`](theory/): what is already known about computing and learning with large numbers, and what a learner's
  form (search, network or gradients) does and doesn't decide
- [`history/`](history/): the early experiments, write-ups and records ([history/README.md](history/README.md)); six
  early folders stay at the top level because the code above still loads them: `anchors/`, `general_lessons/`,
  `learned_remainder/`, `library/`, `litmus/` and `wider_menus/`
