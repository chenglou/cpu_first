# Grading the deep file against the contest's archive

On 2026-10-28, a friend who judges the contest pointed us to the archive of every submission. This folder uses it to
grade the deep file, [integration/legal_deep/submission_stage3_deep.py](../integration/legal_deep/submission_stage3_deep.py)
(md5 `56acca73`), three ways:
1. its score against the leaderboard's;
2. whether our runner grades like the hosted evaluator;
3. a blind rule review next to the top three.

All of it is unofficial: the contest closed on 2026-08-31. The runs cost about $21.

## Sources

- **The organizers' retrospective**, [mcleish7/one-layer-deeper-retrospective](https://github.com/mcleish7/one-layer-deeper-retrospective)
  (2026-09-08). It covers what the three leaders did, Hard's hidden twist (N's decimal digits reversed in the prompt),
  and why the organizers named Yash Kant the winner.
- **The archive**, [GPUMODE/one-layer-deeper-submissions](https://huggingface.co/datasets/GPUMODE/one-layer-deeper-submissions)
  (snapshot 2026-09-07).
  - It holds all 15,602 accepted uploads, each with its `submission.py` and the `result.json` the hosted evaluator
    stored.
  - Its README and the retrospective call it private, but on 2026-10-28 Hugging Face listed it as public and ungated.
  - Its `data/hard.parquet` matched the archive's checksum.
- **Participant files keep their own terms** ("No blanket license"), so none are copied here.
  - [extract.py](extract.py) writes chosen uploads to `sources/`.
  - `raw/` holds our runner's records, which embed the source.
  - Both folders are gitignored.

## 1. The score

| Entry | Certified T: seen / unseen N | Correct: test, ood_t, ood_n_t | Training used | Model state |
|---|---|---|---|---|
| CodeReclaimers `d83efe07` (rank 1) | 64 / 64 | 9999/9999, 10002/10002, 10002/10002 | 3,600 s | 313,886,637 |
| Yash Kant `86e9acf2` (rank 2) | 64 / 64 | all | 3,601 s | 66,137 |
| Sahil Verma `9fa00118` (rank 3) | 64 / 64 | all | 3,600 s | 254,149 |
| alirezashirvani-jr `f5d75083` (rank 4) | none | 5, 3108, 1 | 3,600 s | 195,997,691 |
| **The deep file** `56acca73` (official record) | 64 / 64 | all | 519 s | 1,019,008 |

Only 3 of the 146 ranked entries certified any rung, according to the archive's own statistics; every other one failed
at T=1. The deep file ties those three on both ranked columns. With no uncertified rung left to compare, the
leaderboard falls back to submission time, so it would place fourth. Its seeds 1–7 certify too: 8 of 8, with training
from 502 to 741 s.

## 2. Does our runner grade like the hosted evaluator?

**Setup.**
- Five uploads were rerun byte for byte: each one's md5 equals the organizers' `db_md5`.
- They ran through [tools/modal_runner.py](../tools/modal_runner.py) with the untouched official manifest, on
  `h100_sealed`: an H100 with no network and no access to other Modal resources. Each record confirms the network was
  blocked.
- Each file was checked first. All six extracted files import only torch, math, time, random, re, collections,
  fractions, itertools, contextlib and the contest's `benchmark` API, and all pass the contest's own validator.
- [compare.py](compare.py) compares each rerun with the stored result and writes [calibration.json](calibration.json).

| Upload | Stored | Rerun | Same |
|---|---|---|---|
| `d83efe07` CodeReclaimers | 64/64, 100% | 64/64, 100% | every field |
| `86e9acf2` Yash Kant | 64/64, 100% | 64/64, 100% | every field |
| `9fa00118` Sahil Verma | 64/64, 100% | 64/64, 100% | every field |
| `309222da` Maxwells-Demons (rank 6) | none; 106, 4, 4 correct | none; 106, 4, 4 correct | every field |
| `f5d75083` alirezashirvani-jr (rank 4) | none; 10.38% | none; 10.91% | T only: its T=1 rung got 91 of 768, against 98 |

"Every field" means four things match: certified T on both profiles, all three splits' correct counts, all 14 rungs'
counts, and the model-state size.

**Results.**
- Four of five agree exactly, a partial scorer included.
- The fifth differs within the run-to-run noise of a file that trains for a fixed hour. It ran 0.86 times the hosted
  run's steps.
- All five read the same data as the deep file's records (data md5 `a6414ec9`). So the released H1 generator and our
  runner reproduce the hosted grading.

**Speed.** Steps per hour differed from the hosted runs by 0.86 to 2.44 times, so wall times depend on the host. Suppose
a host were as much slower as the largest ratio seen. Then the deep file's slowest seed would train in about 1,810 s
instead of 741, still inside the hour.

**Cost.** The five runs cost about $21 at list price. A 60-second sealed smoke test of the contest's baseline
([plans/smoke_sealed.json](plans/smoke_sealed.json)) checked the seal first, for $0.15.

## 3. A blind rule review, next to the top three

**Setup.**
- Four reviewers (Claude agents, one file each) got the same brief: the rules, the task and its hidden twist, and the
  judges' central question.
- That question: how much of the representation, the per-step transition and the control across T is learned from the
  supplied labels? The alternative is that it's supplied by the author, or fit to targets the code makes itself.
- The files were renamed A–D, and no reviewer knew whose file it had or what the organizers concluded.
- [reviews/](reviews/) holds the four reports, condensed.

| File | Class | Learned (1–10) | Rule 7 | Rule 8 | Rule 9 | Rules 12/14 |
|---|---|---|---|---|---|---|
| A: Sahil Verma `9fa00118` | (a) hard-codes the target algorithm | 2 | violation | violation | minor concern | violation |
| B: **the deep file** `56acca73` | (b) uses the algorithm to assist learning | 2 | concern | violation | minor concern | violation |
| C: CodeReclaimers `58311e4a`, the version the organizers reviewed | (a) | 2 | violation | violation | violation | violation |
| D: Yash Kant `86e9acf2`, the organizers' winner | (a) | 2 | violation | violation | violation | violation |

**The reviewers against the judges.** For each leader, the reviewers found what the retrospective found:
- C computes the law it selects, and fits on the CPU;
- A runs supplied exact programs;
- D's arithmetic procedure is hand-wired, and its digit cells are fit to targets computed on the CPU.

Reading the rules strictly, the reviewers put all three in class (a). The organizers weighed the same facts and still
named D the winner, because its network "learns reusable digit operations". So the reviewers are stricter than the
judges, and they applied the same standard to all four files.

**On the deep file**, the reviewer found no line that computes x^(2^T) mod N:
- the tables start random;
- the book is empty;
- the program is discovered, and kept only if it reproduces the labels exactly, at T=1 and for larger T.

That is why it's the only one of the four in class (b). Against it, the reviewer listed:
- the author supplies the digit-wise arithmetic unit, the pointer and the count loop over T;
- the residue rule and `law_target` define the practice for comparison, addition, multiplication and reduction, and
  the residue rule's x + d·N operands count as augmentation;
- formulas are read off the labels by search;
- training returns zero logits;
- the searches run inside the training forward pass.

**These are the questions [REVIEW.md](../REVIEW.md) §5 leaves to a judge**: rules 8 and 14 first, then rule 7. The
reviewer ruled against the entry on each, but found no fact that REVIEW.md doesn't state.

We checked one claim by hand: the reviewer's "hand-written doubling reduction" that makes the law reader's targets
(lines 2842–2851). It is a designed procedure, but every step runs the learner's own routines (`lt`, `add`, `red`). So,
as REVIEW.md's rule 14 says, no target comes from Python's arithmetic.

## 4. Where that leaves the deep file

- **Score.** It's at the top with the three leaders, and it trains in about a seventh of their hour. Our runner
  reproduces the hosted evaluator's grading.
- **The rules read strictly.** No top entry is learned and clear, the deep file included. The deep file is the only one
  not judged to hard-code the target.
- **The judges' reading.** They chose a winner whose procedure is hand-wired, for learning its digit operations. By that
  reasoning the deep file is a contender:
  - it also learns its digit tables: from random, on the GPU, with no targets from Python arithmetic;
  - it also composes them over T;
  - unlike the winner, it finds the procedure by search instead of having it wired.

  This judgment is ours, not a reviewer's or a judge's.
- **Limits.** There was one model reviewer per file. The judges may weigh things differently. All of it is unofficial.

## Reproduce

Download `data/hard.parquet` (9.7 MB) from the archive into any folder, and check it against the archive's
`checksums.json`. Then, from `cpu_first/`:

```bash
PY=~/github/sotaku/venv/bin/python
$PY -I grading/extract.py PATH/TO/hard.parquet d83efe07 86e9acf2 9fa00118 f5d75083 309222da 58311e4a
$PY tools/modal_runner.py --mode plan --plan grading/plans/calibrate.json --out grading/raw   # about 75 min, five H100s
$PY tools/modal_runner.py --mode fetch --run TAG --wait
python3 grading/compare.py
```
