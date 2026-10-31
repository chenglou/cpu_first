# Addendum 2: the whole runs again, with the anchors kept to the plain search's finds

Written 2026-10-30, after the fifth launch's outcomes (GATE.md §6) were seen and before any run of this addendum was
launched. GATE.md §1 records its hash.

## Why

- In the fifth launch, the gate's second-search finds took the anchor places of anchored pairs (`anchor_cost: 2`: every
  scored pair holds one of the two costliest learnable problems).
- The outcomes changed both ways. The triangular numbers under alloc_deep came (round 3), and (x + 1)² came a round
  early. But four wins came later, and four did not come in 5 rounds.
- GATE.md §6 traces the losses to the anchor places. In the twins of the four round-0 wins, the pair that found the
  target was anchored on x mod N (`law/mod(X,N)/x+qN`). With the gate's cost-7 finds as anchors, that pair was never
  scored.
- This rerun separates the gate's own effect (its finds practised, paired and picked) from that interaction.

## What changes

- One line of the learner: anchored pairs anchor only on the two costliest learnable problems that the plain search
  solved. `learn_anchor.py` is written by `make_learner.py` from `consolidate/learn_newton.py` with that one asserted
  replacement, under `cfg["anchor_plain"]`.
- A problem that only the gate's second search solved stays learnable. It can still be a pair's other member, a single,
  or a pick.
- Nothing else changes: the same 20 runs, gates and second-search cap, at most 5 rounds. The groups `gt2_a/b/c` hold the
  runs of `gt1s_a/b/c`, on 3 containers of 16 CPUs with 8 plain-search workers and 8 gate workers each, as in the fifth
  launch.
- Each container starts from copies of its fifth-launch group's stores, as after a preemption: a search already done
  replays its record, and the rest is searched. So round 0 up to the pair looks is the fifth launch's exactly (the probes
  do not depend on the anchors), and every difference comes from the anchor rule.
- The plain twins stand: the consolidation's records, and the plain reruns `pt1_a/b` of the eight non-wins.

## Round 0's anchors, known before the launch

From the fifth launch's round-0 records:

| formula | config | anchors in round 0 | against the twin's |
|---|---|---|---|
| 2x + 1 | deep | `restrict/remainder[L.X[INC.X] -> X]/1<X`, `restrict/remainder[L.X[INC.X] -> X]/X+X<X·N` | differs: the twin's `…/N<X+X` is a duplicate of a gate find here |
| x³ | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | same |
| Fibonacci | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | differs: `law2/min(monus(N,X),X)` is the gate's find here |
| floor(x²/N) | deep | `restrict/remainder[DEC.X L.N[MOV.A.N MOV.N.X] -> A]/X·X<X+X`, `law/mod(X,N)/x+qN` | no twin |
| x² mod N | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | differs: `law2/min(monus(N,X),X)` |
| Lucas | deep | `law2/absdiff(mul(X,N),N)`, `law/mod(X,N)/x+qN` | no twin |
| x² − 1 | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | differs: `law2/min(monus(N,X),X)` |
| x² + 3 | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | differs: `law2/min(monus(N,X),X)` |
| 2^x | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | same |
| x² + 1 | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | differs: `law2/min(monus(N,X),X)` |
| 10·x² | deep | `law2/eq(mul(X,X),N)`, `const/10->A` | differs: `law2/min(monus(N,X),X)` and `law2/le(mul(X,X),N)` |
| 3·x² | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | differs: `law2/min(monus(N,X),X)` |
| triangular | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | no twin |
| Tribonacci | deep | `restrict/remainder[DEC.X L.X[MOV.A.X INC.X] -> X]/X·X<X·N`, `law/mod(X,N)/x+qN` | no twin |
| (x + 1)² | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | same |
| x² + x | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | same |
| x³ + x | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | no twin |
| x⁴ | deep | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | differs: `law2/min(monus(N,X),X)` |
| 10·x² + 1 | ns | `const/11->A`, `const/11->T` | differs: `law2/min(monus(N,X),X)` |
| triangular | ns | `law/mod(X,N)/x+qN`, `law2/lt(succ(X),N)` | same |

- x mod N is an anchor in all 11 runs whose twin's program calls it.
- The anchors are the twin's in 5 of the 15 runs with a twin. In 9 of the other 10, the twin's costliest anchor
  `law2/min(monus(N,X),X)` (cost 7) is the gate's find here. The twin's allocator found it with the 10M deep probe;
  here the gate's second search found it after the 1M probe, and the allocator never deep-probes a learnable problem.
- So the anchors are the gated run's two costliest plain finds, not always the twin's. In every win, the twin's best
  round-0 pair is scored again: its members are learnable here, and one of them is an anchor.

## Predictions

| measure | predicted | confidence |
|---|---|---|
| x mod N in round 0 | picked in at least 10 of the 11 runs whose twin picked it | medium |
| x² mod N, x² − 1, x² + 1, x² + x | all four come in round 0, as in their twins. Their twin's pair found the target in its look and gained 66–91 bits; no gate find gains more than 28 bits alone in these runs | medium-high |
| 3·x², x³, 2^x, triangular (alloc_ns) | all four come within 5 rounds, and at least two of them in round 1, as in their twins. In three of these runs a gate find gains 32–45 bits alone, so it will often be picked beside x mod N, and the later rounds can differ from the twins' | medium-low |
| 2x + 1, Fibonacci, x² + 3 | the same round as their twins (0, 1, 1) | medium-high |
| (x + 1)² | round 0, as in the fifth launch: the gate's remainder routine gains 74 bits alone, and it can still be picked, now beside a plain anchor | medium |
| triangular (alloc_deep) | does not come in 5 rounds | medium-low |
| the other seven non-wins | none comes in 5 rounds | high |
| overall | the outcomes move back to the twins': at most two of the eight late or lost wins are still late or lost | medium |

## What would refute them

- **The round-0 account (GATE.md §6):** refuted if x² mod N, x² − 1, x² + 1 or x² + x does not come in round 0, although
  the pair that found its target is scored again.
- **The anchor places as the main route of the losses:** refuted if three or more of the eight late or lost wins
  (x² − 1, x² + x, 2^x, the triangular numbers under alloc_ns; x² mod N, x² + 1, 3·x², x³) are still later than their
  twin's round, or missing after 5 rounds. Then the gate's finds change the outcomes as partners and picks too, not
  only as anchors.
- **The invention:** if the triangular numbers under alloc_deep still come, the invention is the gate's own effect (its
  finds picked as partners), not the anchors'. If they do not come, it came through the anchor places.

## Limits

- A soft $100 inside the study's $1,000 cap.
- `integration/ablation_s3/spend.py 5750` before the launch, and one watcher on the study's cap and the monthly guard.
- At most 8 containers (3 used). The watcher and the apps stop when the runs end.
