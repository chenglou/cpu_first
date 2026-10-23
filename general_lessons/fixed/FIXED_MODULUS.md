# One fixed modulus: the answers choose whole digit codes

## In plain words

<!-- PLAIN -->
**The question.** The general file ([`submission_general_v2.py`](../submission_general_v2.py), LESSONS.md §11) learns
H1 and seven other formulas with one unchanged file. It still learned nothing on the 14 practice datasets that use one
fixed modulus (at most 1.8%). The reason: the model works out which digit token means which value, and which way N and
x are written, from one fact, "x is smaller than N". With many moduli that fact compares every pair of digits somewhere.
With one modulus it only ever compares a digit with one of N's own digits, so thousands to millions of digit codes fit
every fact. The subtraction table, which gets its meaning from the same fact, never became exact either. The answers do
carry the missing information, but the file never searched whole codes. This study makes it do that:
[`submission_general_v3.py`](../submission_general_v3.py).

**What v3 changes.** Everything new switches on only where the data leaves something open. On H1 the facts pin the
code after the first batch, so v3 is v2 there, step for step.
- **The reader keeps its digit-order facts.** Every pair of digit tokens that "x < N" orders, for every way of reading
  the prompt. From them it counts, exactly, how many whole codes each reading leaves.
- **The reader's ranking teaches the table's borrows.** When the facts leave the code open, the table also learns from
  pairs of numbers taken from different rows: "a − b borrows exactly when a < b", where the reader's own ranking says
  which is smaller. The reader's labels are ranks, so this teaches ordinary subtraction on the labels, whichever code
  the reader happens to hold. The machine becomes exact before the code is right, and can then test whole codes.
- **A tournament over whole codes.** Once the machine is exact, every whole (reading, code) the facts leave is a
  contestant, together with every lesson set and every option of the round menu. Each step, every surviving
  contestant runs the machine on the batch's questions, read through its own code, and is checked against the answers
  read the same way. One wrong answer and it is out. The winner becomes a lesson for the reader, which then learns
  it by gradient like any other fact.
- **Guards learned at full scale.** Until the answers choose, the writer learns no code. Once they choose, the
  chosen reading pulls the pointers with its own strength, with their shared shape held. Without these, things learned
  under the provisional code resisted being undone (§5).
- **One modulus also starves the multiply.** When every row has the same N, the multiply's column rules are scored
  whole, exactly, like the code. Its blend of rules had stalled on short numbers (§4, §5).

**Tiny first: what each piece does on its own.**
- **The census** (exact counting on the training rows): the facts leave 16,800 (M2) to 2,963,520 (M8) whole
  candidates. The answers single out one in 13 of the 14 datasets, the true one. That includes M1 and M2, which an
  earlier count called hopeless: it read only answers with T ≤ 2, and those datasets have none. Reading the answers
  at the smallest T present, as v2 does, settles them. The 14th is E1: its N = 323 reads the same both ways, so the
  answers cannot say which way N is written. The code itself is still pinned there.
- **The table on one modulus.** Without the new lesson the table never became exact on one modulus (0 of 24 small
  runs). With it, it became exact in all 24, under the true code and under wrong codes alike. With many moduli it
  changes nothing.
- **The tournament**, run with v3's own code on real training batches and an exact machine: on every one of the 14
  datasets, one batch of answers left only the true code. The same batch also chose long division and the plain square
  from the 80 combinations of lesson set and round options. It cost 4,754 to 252,968 runs of the machine (4 to 66 s on
  a laptop CPU). Using the tournament's soft weights instead (TOURNAMENT.md: a learned weight per contestant, losers
  dropped once far behind), the same survivors were left after 13 steps.

**Full runs, zero per-task changes.** <!-- FULL_SUMMARY -->
On the final build **all 14 fixed-modulus datasets learned**, with the file byte for byte on each (relaxed budget,
seed 74, B200s): 100% on each dataset's own test and out-of-distribution splits. In every run the census and one batch
of answers chose the true code, at steps 337 to 1,071, from 16,800 to 2,963,520 contestants, with 2,555 to 116,791
machine runs. E1, E10, M1, M2, M7 and M8 also certified T = 64 on both depth ladders, unseen moduli included. The other
eight certified up to T = 8: their training T always has one digit, nothing tells "16" from "61" (LESSONS.md §4), and
their readers kept the direction they started with, backwards here. E1's tie of readings came out the right way.
<!-- SEEDS_FIXED -->The extra seeds learned as well: E10 on seeds 1 and 2, which had stalled on the previous build, and M2 on seed 1. That is 17 of 17 fixed-modulus runs of the final build.<!-- /SEEDS_FIXED -->

Getting there took four builds. The first learned E1, M1 and M7, and showed where provisional choices resist being
undone; the second and third fixed that and reached 13 of 14 (§5). The last failures were never the code. They were
the multiply's column rules, learned as a blend, stalling on short numbers. Scoring those rules whole, like the
code, fixed it.
<!-- /FULL_SUMMARY -->

**H1 and the other tasks.** <!-- H1_SUMMARY -->
Yes. Through its untouched official manifest on one H100, the final file certified H1: every one of 768 questions right
at every T up to 64, on seen and unseen moduli, every split exact, in 1,191 s of training. It is v2's official run
step for step: H1's facts pin the code after the first batch, so none of the new machinery ever switches on. 8 of 8
seeds certified too (B200s). <!-- OTHER_PLAIN -->The other formulas still pass (rho, ten, cube, Montgomery's, and the held-out 10·y² + 1, y² + 3 and the reordered prompt), and so do the sampled-moduli tiers E4, M3, M4 and M5. E3 and E5 fail on seed 74 exactly as v2 does, line for line: the general file's known slow seed. y² − 1 is still outside the menu.<!-- /OTHER_PLAIN -->
<!-- /H1_SUMMARY -->

**What this says.** <!-- SAYS -->
LESSONS.md ended on "the answers say [what the digits mean], but only to a model that searches whole digit codes,
which this design does not do." v3 does, and the missing information was all there: one batch of answers singles out
the true code among up to 3 million candidates. Four things follow.
- **The meaning of a digit had to live in one place.** The table's borrows and the reader's code both drew their
  meaning from x < N. With one modulus neither could settle, and each waited for the other. Tying the borrows to the
  reader's ranking put all of the meaning in the reader, where the answers could choose it whole. The machine then
  learns ordinary arithmetic on labels under any code. That is a designer's decision about what a borrow means, and
  the rules table says so.
- **Provisional choices have to stay revisable.** Three of the failures at full scale had one shape: something
  learned for hundreds of steps under a provisional choice had to be undone once the answers chose. The writer's code
  was clipped below 10^-6. The reading was held by a floor of 10^-6. The pointers' shared shape was moved by a reading
  lesson on the writer's behalf. The fixes either keep a lesson from running before the choice (the writer's code),
  or give the choice a lesson whose strength doesn't depend on how far the old choice went (the chosen reading's
  log-probability, with the shared shape held, as v2's own T lesson does).
- **Whole options, again.** The other failures were the multiply's layout: the last part of the machine still learned
  as a blend of options on the wire. One modulus gives it short, repetitive numbers, and the blend stalled on rules
  that drop digits. Scoring the column rules whole, as the code and the chain are scored, left exactly the true rules
  every time. This is the project's oldest lesson (OVERVIEW.md §3), met once more.
- **Some ties no answer breaks.** E1's N reads the same both ways, so no E1 question can tell which way N is written;
  the answers pinned the code and kept the reader's own reading, the right one here. On data whose T always has one
  digit, nothing can tell "16" from "61". That coin flip is unchanged from v2 and decides only the depth ladders past
  T = 8, not any dataset's score. What would break these ties is a prior, not data: read every number the way x is read
  (x's direction is pinned on every dataset), and read T the way the answer is written (LITMUS.md).

The limits are plain:
- The hypothesis space is every bijection of ten digit tokens to ten labels. The census enumerates all 10! of them,
  fine for base 10, but a larger alphabet would need a constructive enumeration (the linear extensions directly, or a
  beam over partial codes).
- The relaxed budget is still needed: on a B200 the code was chosen <!-- CHOSE_RANGE -->252 to 1,093 s<!-- /CHOSE_RANGE -->
  into training, far past Easy's 60 s, and Medium's 600 s on an H100 would cut most runs before the choice.
- Each dataset ran on one seed, as LESSONS.md ran the tiers, plus extra seeds for E10 and M2.
- The knowledge moved once more, as LESSONS.md found for the formulas: the designer now writes the tie, the hypothesis
  spaces (codes, column rules) and the scoring's staging, the same on every dataset, and the data chooses within them.
<!-- /SAYS -->
<!-- /PLAIN -->

## Details

### 1. The census: what the facts leave, and what the answers leave

[`code_census.py`](code_census.py) counts on every dataset's training rows, with exact arithmetic on the labels
standing in for an exact machine, as [`../code_search.py`](../code_search.py) did. It differs from that count in two
ways, so that its numbers are the model's own:
- **The reader's semantics.** A reading hypothesis says which digit run is N and which way it is read, and the same
  for x (T is the run left over). It passes a row when, at the highest place where the two token strings differ, x's
  token ranks below N's. Where x has ended there, N's token must not be the one coded 0. code_search.py padded x with
  the true 0 token instead, which forbids more codes. Every reading is counted, not only N as written.
- **v2's answer rows**: rows with one-digit T up to max(2, the smallest one-digit T in the data). code_search.py used
  T ≤ 2, which left M1 and M2 (T = 4, 8, 16) without a single answer row.

<!-- CENSUS -->
| task | N | training rows | readings the facts leave (N's run and direction, x's: codes) | whole candidates | code_search.py's count (N as written, x padded with the 0 token) | answer rows (v2's rule) | candidates that reproduce every answer row | the true one among them |
|---|---|---|---|---|---|---|---|---|
| e1 | 323 | 600 | 0011: 302,400, 0111: 302,400 | 604,800 | 302,400 | T <= 2: 400 | 2 (0011 + true code, 0111 + true code) | yes |
| e2 | 899 | 1,920 | 0010: 45,360, 0011: 403,200, 0111: 45,360 | 493,920 | 40,320 | T <= 2: 1,280 | 1 (0111 + true code) | yes |
| e6 | 247 | 588 | 0011: 1,209,600, 0111: 51,840 | 1,261,440 | 34,560 | T <= 2: 294 | 1 (0111 + true code) | yes |
| e7 | 287 | 528 | 0011: 1,209,600, 0111: 28,800 | 1,238,400 | 16,800 | T <= 2: 352 | 1 (0111 + true code) | yes |
| e8 | 287 | 528 | 0011: 1,209,600, 0111: 28,800 | 1,238,400 | 16,800 | T <= 2: 352 | 1 (0111 + true code) | yes |
| e9 | 299 | 600 | 0010: 181,440, 0011: 1,209,600, 0111: 181,440 | 1,572,480 | 120,960 | T <= 2: 400 | 1 (0111 + true code) | yes |
| e10 | 403 | 792 | 0111: 241,920 | 241,920 | 241,920 | T <= 2: 528 | 1 (0111 + true code) | yes |
| m1 | 10403 | 27,000 | 0011: 1,814,400, 0111: 151,200 | 1,965,600 | 90,720 | T <= 4: 9,000 | 1 (0111 + true code) | yes |
| m2 | 38021 | 81,000 | 0111: 16,800 | 16,800 | 16,800 | T <= 4: 27,000 | 1 (0111 + true code) | yes |
| m6 | 1517 | 3,510 | 0011: 1,814,400, 0111: 81,000 | 1,895,400 | 81,000 | T <= 2: 2,340 | 1 (0111 + true code) | yes |
| m7 | 1763 | 4,050 | 0011: 1,814,400, 0111: 13,320 | 1,827,720 | 8,100 | T <= 2: 2,700 | 1 (0111 + true code) | yes |
| m8 | 1333 | 2,970 | 0010: 362,880, 0011: 1,814,400, 0111: 786,240 | 2,963,520 | 453,600 | T <= 2: 1,980 | 1 (0111 + true code) | yes |
| m9 | 1927 | 4,185 | 0011: 1,814,400, 0111: 18,648 | 1,833,048 | 10,800 | T <= 2: 2,790 | 1 (0111 + true code) | yes |
| m10 | 1739 | 3,780 | 0011: 1,814,400, 0111: 17,820 | 1,832,220 | 10,710 | T <= 2: 2,520 | 1 (0111 + true code) | yes |
<!-- /CENSUS -->

- **Readings.** 0111 is the truth: N is the first run, x the second, both read from their right ends (LSB at the
  right, the usual way). 0011 reads N from its left end (backwards), which is how H1 writes N. 0010 reads both N and x
  backwards. Reading N backwards passes the facts in 12 of the 14 datasets, with up to 1,814,400 codes, because a
  backwards N is still larger than every x under many codes. The answers rule out every one of them except E1's.
- **E1's tie** is real and harmless for E1's own score: 323 reversed is 323. A model reading N backwards computes
  the same numbers on every E1 question, but would misread an unseen modulus on the depth ladder's "unseen moduli"
  rungs, which are diagnostics, not part of the score.
- **T's direction** stays a separate tie, as in LESSONS.md §4. On data whose T always has one digit, nothing tells
  "16" from "61". Only M1 and M2 train on a two-digit T (16), and there v2's own T lesson decides it.

### 2. The table on one modulus: the reader's ranking ties the borrows

[`tiny_tie.py`](tiny_tie.py) is [`../tiny_table.py`](../tiny_table.py) (the subtraction table alone, learned from a
random start by the anchors' lessons, W = 6, the reader given) with two options:
- `tie = 1`: sub.order on a read number and another row's (x against the next row's x), on every top part, both ways
  round. The premise "a < b" is the reader's own comparison of the two label strings (the miner's `compare`: the
  highest place where they differ decides). Its hard verdicts join the facts the digit lessons wait for, and the
  entries its subtractions read join the order laws' entries;
- `code = wrong`: the reader reads with a random wrong code that still passes x < N on every row, so every number is
  a label number. The table is then scored against the standard table on the labels (r − m − c).

<!-- TIE -->
| modulus | the table's borrow lessons | the reader's code | seeds exact | step exact | entries right at the end (digit, borrow, both) |
|---|---|---|---|---|---|
| 323 | the facts only | true | 0/3 | -, -, - | 36/124/34, 45/145/35, 43/126/34 |
| 323 | the facts only | a wrong one that passes x < N | 0/3 | -, -, - | 34/148/30, 38/133/31, 49/141/41 |
| 323 | the reader ties the borrows | true | 3/3 | 124, 108, 117 | 200/200/200, 200/200/200, 200/200/200 |
| 323 | the reader ties the borrows | a wrong one that passes x < N | 3/3 | 125, 108, 114 | 200/200/200, 200/200/200, 200/200/200 |
| 899 | the facts only | true | 0/3 | -, -, - | 41/161/36, 35/161/32, 60/153/55 |
| 899 | the facts only | a wrong one that passes x < N | 0/3 | -, -, - | 45/145/42, 43/170/41, 51/145/45 |
| 899 | the reader ties the borrows | true | 3/3 | 118, 104, 113 | 200/200/200, 200/200/200, 200/200/200 |
| 899 | the reader ties the borrows | a wrong one that passes x < N | 3/3 | 120, 106, 115 | 200/200/200, 200/200/200, 200/200/200 |
| 1517 | the facts only | true | 0/3 | -, -, - | 45/124/35, 30/146/27, 47/134/38 |
| 1517 | the facts only | a wrong one that passes x < N | 0/3 | -, -, - | 43/118/34, 50/158/47, 52/104/37 |
| 1517 | the reader ties the borrows | true | 3/3 | 225, 87, 106 | 200/200/200, 200/200/200, 200/200/200 |
| 1517 | the reader ties the borrows | a wrong one that passes x < N | 3/3 | 231, 103, 410 | 200/200/200, 200/200/200, 200/200/200 |
| 10403 | the facts only | true | 0/3 | -, -, - | 58/145/49, 38/161/36, 51/139/45 |
| 10403 | the facts only | a wrong one that passes x < N | 0/3 | -, -, - | 33/130/32, 45/140/39, 39/137/36 |
| 10403 | the reader ties the borrows | true | 3/3 | 98, 76, 78 | 200/200/200, 200/200/200, 200/200/200 |
| 10403 | the reader ties the borrows | a wrong one that passes x < N | 3/3 | 154, 141, 120 | 200/200/200, 200/200/200, 200/200/200 |
| many (4-5 digits) | the facts only | true | 3/3 | 77, 73, 71 | 200/200/200, 200/200/200, 200/200/200 |
| many (4-5 digits) | the reader ties the borrows | true | 3/3 | 76, 73, 71 | 200/200/200, 200/200/200, 200/200/200 |
<!-- /TIE -->

**Why it works, and what it costs in purity.** The reader's labels are ranks among the digit tokens: label k means
"the k-th smallest token". Under any code, then, "a − b borrows exactly when a < b" teaches ordinary subtraction on the
labels. With many moduli the facts already order every pair of digits, so the tie is redundant there (the last two
rows). With one modulus they order only N's digits, and the tie supplies the rest. It is a design choice. It fixes the
borrow half of the table's meaning to the reader's ranking by construction (LESSONS.md §8, open question 1, proposed
exactly this), so the code's meaning lives entirely in the reader, where the answers can choose it. The comparison
itself is the one the reader's Bradley–Terry lesson and the miner already used. The tie adds no arithmetic, only new
pairs to compare.

**The consequence the tournament needs.** Under a wrong code that passes x < N, the table learns exactly the same
standard label table as under the true one (every run above). So the machine can be exact, on every label-free check,
before anyone knows the code. Then the answers can score whole codes through it.

### 3. The code tournament

[`tiny_tournament.py`](tiny_tournament.py) runs v3's own census and scoring (`digit_facts`, `census`, `code_score`,
`code_lesson`) on each dataset's real training batches, with every table and switch at the truth (MODE = "designed"),
so that only the tournament is under test. The reader's digit-order facts are gathered on 12 batches first, as in
training. The selector and the round menu start from the file's random start, so the code must be found jointly with
the lesson set and the round options: 2 sets × 40 combinations.
- **hard** (v3 as built): a (contestant, lesson set, round) triple that fails an answer row is out. The survivors' sets
  and round options train the selector and the round menu.
- **soft** (TOURNAMENT.md's form): a learned weight per contestant (Adam at LR 0.5), the loss −log of the passing
  triples' total weight, and contestants more than 12 nats behind the best no longer run.

<!-- TOURN -->
| task | weights | contestants (census) | contestants left | batch it ended at | reading chosen | code chosen | (set, round) left after batch 1 | machine runs | s (laptop CPU) |
|---|---|---|---|---|---|---|---|---|---|
| e1 | hard | 604,800 | 2 | 1 | 0011 | true | below + one/0/none | 4,754 | 4.0 |
| e2 | hard | 493,920 | 1 | 1 | 0111 | true | below + one/0/none | 14,193 | 7.4 |
| e6 | hard | 1,261,440 | 1 | 1 | 0111 | true | below + one/0/none | 34,444 | 10.6 |
| e7 | hard | 1,238,400 | 1 | 1 | 0111 | true | below + one/0/none | 30,658 | 9.7 |
| e8 | hard | 1,238,400 | 1 | 1 | 0111 | true | below + one/0/none | 30,658 | 9.6 |
| e9 | hard | 1,572,480 | 1 | 1 | 0111 | true | below + one/0/none | 6,849 | 6.1 |
| e10 | hard | 241,920 | 1 | 1 | 0111 | true | below + one/0/none | 9,431 | 5.7 |
| m1 | hard | 2,131,920 | 1 | 1 | 0111 | true | below + one/0/none | 252,968 | 65.5 |
| m2 | hard | 33,600 | 1 | 1 | 0111 | true | below + one/0/none | 32,396 | 9.1 |
| m6 | hard | 1,895,400 | 1 | 1 | 0111 | true | below + one/0/none | 37,026 | 13.6 |
| m7 | hard | 1,827,720 | 1 | 1 | 0111 | true | below + one/0/none | 208,165 | 40.9 |
| m8 | hard | 2,963,520 | 1 | 1 | 0111 | true | below + one/0/none | 7,674 | 6.0 |
| m9 | hard | 1,833,048 | 1 | 1 | 0111 | true | below + one/0/none | 206,663 | 42.2 |
| m10 | hard | 1,832,220 | 1 | 1 | 0111 | true | below + one/0/none | 206,770 | 41.3 |
| e1 | soft | 604,800 | 2 | 13 | - | - | - | 72,288 | 52.1 |
| m2 | soft | 33,600 | 1 | 13 | 0111 | true | - | 421,013 | 118.6 |
| m8 | soft | 2,963,520 | 1 | 13 | 0111 | true | - | 97,464 | 73.6 |
<!-- /TOURN -->

- **One batch suffices.** In every dataset the first batch of answers (the rows with one-digit T in 512 random
  training rows) left the true code alone, or with its palindromic twin in E1. It also left a single (set, round)
  pair: long division with the plain square.
- **Why it is cheap.** The machine runs once per distinct input, and rows go one at a time, those with the fewest
  distinct digit tokens first. The first row's answer then depends on the codes of only 3 to 6 tokens, so millions of
  contestants share a few thousand inputs. That is up to 2,963,520 contestants × 80 (set, round) combinations, about
  2.4·10^8 triples, scored with 4,754 to 252,968 machine runs. The verdicts are the ones a run of every triple on every
  row would give.
- **Hard or soft.** With the machine exact, a failed answer is final, so v3 drops the triple at once. The soft form
  reaches the same survivors at step 13. It needs 12 more steps, each re-scoring every contestant still in the
  running (72,288 to 421,013 machine runs in all, against 4,754 to 32,396 for the same datasets), and a learned weight
  per contestant. Its robustness to wrong verdicts would matter only while the machine can still be wrong, and v3
  waits until it cannot.

### 4. v3: the file

[`submission_general_v3.py`](../submission_general_v3.py) is built by [`make_v3.py`](make_v3.py) from v2 (md5
`30d15786…`, never modified; each patch must match exactly once), so the diff is that file. Constants `FIXED_CODES`
(True), `CODE_ROWS` (256), `CODE_BATCH` (64) and `LAYOUT_CENSUS` (64). No new parameter.
- `digit_facts` (from io stage 3, inside io_lessons, without gradient): per reading hypothesis, the token pairs the
  x < N lesson orders at the decisive place, the tokens met where x has ended (not coded 0), and whether the hypothesis
  failed a row whatever the code. `code_count` counts each reading's codes exactly, as the linear extensions of the
  orders, with a DP over the 1,024 subsets of tokens. `ambiguous` = more than one whole candidate, once MINE_WIN (10)
  batches have been seen. On H1 the facts pin the code at step 165 (`LR digit_codes pinned_by_facts`).
- While `ambiguous`:
  - the reading mixture credits only the readings the facts leave, and the reading is chosen once the posterior sits
    on them (CHOSEN), not on one of them;
  - the tie lessons (§2) on N, x and T against another row's, from io stage 4, with their hard verdicts in the facts
    the digit lessons wait for and in the table's checks;
  - the writer's code has no lesson. A code learned for the reader's provisional ranking would only have to be
    unlearned: a first build without this gate (`c06097a7`, wave 1 below) never fully unlearned it on E7. The
    writer's pointer lesson covers every slot, continuing the answer's least-significant-first order past its end,
    not only the answer's slots and the one past it: under the pointers' "offsets" shape an untrained slot can land
    on the answer (E2 and E6 in wave 2);
  - `code_lesson`, once every admissible lesson set's I/O lessons have joined and its machine passes its division
    ladder, step law and whole-reduction laws, and the table and the multiply pass theirs. The census enumerates all
    10! codes and filters them by each reading's facts. Each step, the survivors are scored (§3) on the batch's rows
    with one-digit T, each contestant reading x, N, T and the answer through its own code. The loss is the answers'
    mixture over the survivors' (lesson set, round options), with the same factorized weights as v2's selector. If
    nothing survives, the census is made again WIRE_WIN steps later.
- **The choice** (`ctl.leader`): when the survivors share one code (and one reading, or the reader's own reading is
  among theirs), or stop changing for MINE_WIN batches. Then:
  - the chosen order of the digits becomes a Bradley–Terry lesson for that reading's code (all 45 pairs);
  - the reading lesson becomes the chosen reading's own log-probability, field by field, with the pointers' shared
    shape held, as v2's T lesson holds it. The mixture's pull toward a reading is proportional to the weight the
    pointers still give it, which after hundreds of steps on another reading can be too small to move them (M8 in
    wave 1, §5). Unheld, the pull moved every field, the writer included, to another shape (E2 and E6 in wave 2);
  - the writer's code lessons resume.

  The reader learns the code and the reading by gradient, and the prediction path is v2's (`LR code_adopted` when
  the reader reads with them).
- **One N in every row** (a separate gate: every row of the batch has the same N, which H1 never meets once its reader
  has committed to a reading):
  - the multiply's routed ladder also runs on the machine's own square of x (its low 8 digits), alongside x and N.
    With one modulus, half the ladder's numbers are the same number every step;
  - **the multiply's column rules are scored whole** (`layout_lesson`), the code tournament's pattern applied to the
    layout. Once the product cell's own ladder passes (the table already does), every column rule of the menu
    (1,911 per kind) runs the ladder's hard rungs on 64 numbers of the batch (31 x's, N and 32 squares): the low
    digits' rules on rung 1, where every high digit is 0, then the high digits' rules on rungs 1 to 9 with the most
    probable surviving low rule. A rule that fails a rung is out, and the survivors are scored again on every later
    batch. The lesson is −log of each kind's weight on its survivors. On E10's and M2's data, with exact tables, the
    census leaves exactly one rule of each kind, the true ones (4 s on a laptop CPU for all 3,822). The ladder's blend
    of rules stalled on short numbers, where the rules that pile every digit of the multiplicand into one column lose
    too little (§5).

### 5. Full runs: the 14 fixed-modulus datasets

<!-- FULL_NOTES -->
**Wave 1: the first build, and what it taught** (`c06097a7`: the facts, the tie and the tournament of §4, without
its guards for the writer and the reading and without the multiply's changes; six datasets on B200s with the relaxed
budget of 3,600 s and at most 1,500 steps, as LESSONS.md ran the tiers).

The new parts worked at full scale as in the tiny tests. On all five datasets that reached phase C, the census and
one batch of answers chose the true code: M1 at step 430 (1,965,600 contestants, 120,176 machine runs), M7 at 470, M8
at 480 (2,963,520 contestants, 7,214 machine runs), E1 at 605 (the palindrome's two readings, the reader's own kept)
and E7 at 610. **E1, M1 and M7 learned: 100%, and T = 64 on both depth ladders.** They are the first fixed-modulus
datasets this project has learned.

The other three failed for reasons the next build fixes:
- **E7 (21.7%): the writer's code.** The reader read with the chosen code 51 steps after the choice, and the machine
  answered every training answer right. But the writer re-learned only 8 of its 10 values within 60 steps and then
  froze for 800 steps, writing 0 and 2 as the wrong tokens. Its lessons clip probabilities at 10^-6 before taking the
  log, and the provisional code had pushed the right tokens below that, so they got no gradient at all. Fix: until
  the answers choose, the writer's code gets no lesson (its pointer still does). From near its random start it learns
  in a few steps.
- **M8 (0.2%): the reading.** The pointers had committed to reading both N and x from their left ends (reading 0010);
  the answers chose 0111. The reading mixture credits every other reading with a floor of 10^-6, so its pull toward
  the chosen one was that reading's current weight over the floor. That weight is a product of two pointers'
  probabilities, both far down by step 480, and the pointers never moved in the 1,020 steps left. Forcing a choice
  on M7 at step 200 or at step 480, where only N's pointer had to move and less far, flipped it within 40 steps in
  either form (`records/diag/flip480_*`, 100% both). Fix: once the answers choose, the reading lesson is the chosen
  reading's own log-probability, field by field, with no floor.
- **E10 (2.5%): the multiply's layout, not the code.** Phase C never began: the high digits' column rule settled at
  0.65 on column j + 1 for every digit of the multiplicand (the truth is i + j + 1), with the ladder's checks stuck
  at 40%. A run that drops a high digit is right whenever the dropped digit is 0, which on three-digit numbers is
  often. With one modulus, half of the ladder's numbers are the same number every step. Locally on E10 (CPU, the
  same build, `tiny/layout/`): stuck at 40% through step 800 as on the B200. Adding the machine's own square of x to
  the ladder's numbers moved the rule to the truth by step 650. E8 recovered by step 600 either way, so the stall is a
  matter of chance, and the square shifts the odds. Fix: where every row has the same N, the ladder also runs on the
  square of x.

**Wave 2: the second build, and one more lesson** (`32a1d154`: the three fixes above). Its first records:
- **M2 learned**: 100%, T = 64 on both ladders, frozen at step 994. The census came at step 727 (16,800
  contestants, one survivor, 8,271 machine runs), the reader adopted the code at 761, and the writer, learning its
  code only from then, was exact about 35 steps later.
- **E2 (13%) and E6 (12%) failed at evaluation alone.** In training, every check was exact, every answer row right,
  the reader on the true code and reading, the writer exact. Their pointers had to flip, from reading N backwards to
  the chosen reading. The new pull did it, but not by moving the "step" pointer's anchor: it switched the pointers'
  shared shape to "offsets", which reads a 3-digit N from the right end through the N marker (+3, +2, +1). The shape
  is shared with the writer. Under "offsets" each writer slot has its own offset, and slots past a 3-digit answer
  had never had a lesson: slot 7 pointed at the answer's tens digit and wrote a 0 over it. Training never runs the
  writer's pointer, so nothing showed until evaluation (T = 1 rung: 2 of 40). v2's own T lesson holds the shared shape
  for exactly this reason. Fixes: the chosen reading's pull holds the shape too; and, while the facts leave the code
  open, the writer's pointer lesson covers every slot, continuing the answer's order past its end.

**Wave 3: the third build** (`2a9c5ec2`: the two fixes above). Thirteen of the 14 learned on seed 74: E1, E2, E6,
E7, E8, E9, E10, M1, M6, M7, M8, M9, M10. Every run that reached phase C chose the true code and read with it. The
runs that did not learn never reached phase C, and all for one reason: **the multiply's layout**. M2 (on this build's
path; it had learned on `32a1d154`), and E10 on seeds 1 and 2, stalled with the high digits' rule on column j + 1 and
the ladder's checks at 32 to 40%, despite the square of x in the ladder. The layout was the last part of the machine
still learned as a blend of options on the wire, the form OVERVIEW.md's lessons warn against. The final build scores
the column rules whole (§4).

**The final build** (`0805b7d3`: the layout census added). On every run the census of column rules left exactly
the true rules, (1,1,0) for the low digits and (1,1,1) for the high, once the product cell's own ladder passed, and
phase C began 38 to 68 steps later. M2, whose cell was slow on this path, got its census at step 934 and froze at
1,331 with everything exact. <!-- FINAL_NOTES -->All 14 datasets learned on seed 74, and so did the three runs that had stalled before (E10 on seeds 1 and 2, M2 on seed 1): 17 of 17. Every run's census and first batch of answers left the true code alone (E1: with its palindromic twin).<!-- /FINAL_NOTES -->
<!-- /FULL_NOTES -->

**The final build** (`0805b7d3`), seed 74 unless noted, B200s, the relaxed budget:

<!-- TIERS -->
| task | N | build | seed | contestants (census step) | the answers chose | reader read with it at step | score | splits | max T seen / unseen | steps | training s | evaluation s | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 323 | v3 | 74 | 604,800 at step 489 | step 489: reading 0111, code true (2 survivors, a tie of readings) | 536 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1288 | 3602 | 37 | B200 |
| e2 | 899 | v3 | 74 | 493,920 at step 407 | step 407: reading 0111, code true (1 survivor) | 460 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1470 | 3600 | 40 | B200 |
| e6 | 247 | v3 | 74 | 1,261,440 at step 560 | step 560: reading 0111, code true (1 survivor) | 605 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1485 | 3602 | 38 | B200 |
| e7 | 287 | v3 | 74 | 1,238,400 at step 490 | step 490: reading 0111, code true (1 survivor) | 534 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1353 | 3601 | 45 | B200 |
| e8 | 287 | v3 | 74 | 1,238,400 at step 490 | step 490: reading 0111, code true (1 survivor) | 534 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1303 | 3600 | 40 | B200 |
| e9 | 299 | v3 | 74 | 1,572,480 at step 410 | step 410: reading 0111, code true (1 survivor) | 462 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1295 | 3602 | 40 | B200 |
| e10 | 403 | v3 | 74 | 241,920 at step 624 | step 624: reading 0111, code true (1 survivor) | 656 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1280 | 3603 | 44 | B200 |
| e10 | 403 | v3 | 1 | 241,920 at step 578 | step 578: reading 0111, code true (1 survivor) | 609 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1339 | 3600 | 43 | B200 |
| e10 | 403 | v3 | 2 | 241,920 at step 582 | step 582: reading 0111, code true (1 survivor) | 615 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1321 | 3602 | 47 | B200 |
| m1 | 10403 | v3 | 74 | 1,965,600 at step 361 | step 361: reading 0111, code true (1 survivor) | 411 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1536 | 48 | B200 |
| m2 | 38021 | v3 | 74 | 16,800 at step 1071 | step 1071: reading 0111, code true (1 survivor) | 1102 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 2440 | 45 | B200 |
| m2 | 38021 | v3 | 1 | 16,800 at step 582 | step 582: reading 0111, code true (1 survivor) | 616 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1430 | 39 | B200 |
| m6 | 1517 | v3 | 74 | 1,895,400 at step 383 | step 383: reading 0111, code true (1 survivor) | 428 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 3570 | 40 | B200 |
| m7 | 1763 | v3 | 74 | 1,827,720 at step 363 | step 363: reading 0111, code true (1 survivor) | 394 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 3585 | 34 | B200 |
| m8 | 1333 | v3 | 74 | 2,963,520 at step 337 | step 337: reading 0111, code true (1 survivor) | 388 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1380 | 3602 | 37 | B200 |
| m9 | 1927 | v3 | 74 | 1,833,048 at step 404 | step 404: reading 0111, code true (1 survivor) | 447 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1391 | 3603 | 50 | B200 |
| m10 | 1739 | v3 | 74 | 1,832,220 at step 562 | step 562: reading 0111, code true (1 survivor) | 601 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1470 | 3603 | 46 | B200 |
<!-- /TIERS -->

**The earlier builds** (wave 1, `c06097a7`; wave 2, `32a1d154`, stopped once E2 and E6 showed the shape problem):

<!-- TIERS_EARLIER -->
| task | N | build | seed | contestants (census step) | the answers chose | reader read with it at step | score | splits | max T seen / unseen | steps | training s | evaluation s | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 323 | c06097a7 | 74 | 604,800 at step 605 | step 605: reading 0111, code true (2 survivors, a tie of readings) | 646 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 2231 | 21 | B200 |
| e1 | 323 | 2a9c5ec2 | 74 | 604,800 at step 615 | step 615: reading 0111, code true (2 survivors, a tie of readings) | 655 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 2207 | 20 | B200 |
| e2 | 899 | 32a1d154 | 74 | 493,920 at step 538 | step 538: reading 0111, code true (1 survivor) | 576 | 13.2% | ood 12.7%, test 13.8% | 0 / 0 | 1500 | 1915 | 22 | B200 |
| e2 | 899 | 2a9c5ec2 | 74 | 493,920 at step 538 | step 538: reading 0111, code true (1 survivor) | 582 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 1959 | 24 | B200 |
| e6 | 247 | 32a1d154 | 74 | 1,261,440 at step 856 | step 856: reading 0111, code true (1 survivor) | 892 | 13.1% | ood 13.3%, test 12.8% | 0 / 0 | 1500 | 1658 | 22 | B200 |
| e6 | 247 | 2a9c5ec2 | 74 | 1,261,440 at step 856 | step 856: reading 0111, code true (1 survivor) | 894 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1492 | 3603 | 49 | B200 |
| e7 | 287 | c06097a7 | 74 | 1,238,400 at step 610 | step 610: reading 0111, code true (1 survivor) | 661 | 21.7% | ood 20.0%, test 23.5% | 0 / 0 | 1500 | 1924 | 27 | B200 |
| e7 | 287 | 2a9c5ec2 | 74 | 1,238,400 at step 656 | step 656: reading 0111, code true (1 survivor) | 693 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 1875 | 22 | B200 |
| e8 | 287 | 2a9c5ec2 | 74 | 1,238,400 at step 656 | step 656: reading 0111, code true (1 survivor) | 693 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 1897 | 22 | B200 |
| e9 | 299 | 2a9c5ec2 | 74 | 1,572,480 at step 706 | step 706: reading 0111, code true (1 survivor) | 751 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 2854 | 34 | B200 |
| e10 | 403 | c06097a7 | 74 | - | - | - | 2.5% | ood 2.4%, test 2.5% | 0 / 0 | 1500 | 1211 | 2 | B200 |
| e10 | 403 | 2a9c5ec2 | 74 | 241,920 at step 735 | step 735: reading 0111, code true (1 survivor) | 765 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1981 | 21 | B200 |
| e10 | 403 | 2a9c5ec2 | 1 | - | - | - | 0.0% | ood 0.0%, test 0.0% | 0 / 0 | 1500 | 1164 | 125 | B200 |
| e10 | 403 | 2a9c5ec2 | 2 | - | - | - | 1.1% | ood 1.6%, test 0.5% | 0 / 0 | 1500 | 1240 | 367 | B200 |
| m1 | 10403 | c06097a7 | 74 | 1,965,600 at step 430 | step 430: reading 0111, code true (1 survivor) | 487 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1899 | 40 | B200 |
| m1 | 10403 | 2a9c5ec2 | 74 | 1,965,600 at step 446 | step 446: reading 0111, code true (1 survivor) | 489 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1298 | 38 | B200 |
| m2 | 38021 | 32a1d154 | 74 | 16,800 at step 727 | step 727: reading 0111, code true (1 survivor) | 761 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1604 | 41 | B200 |
| m2 | 38021 | 2a9c5ec2 | 74 | - | - | - | 0.1% | ood 0.1%, test 0.1% | 0 / 0 | 1500 | 1626 | 4 | B200 |
| m6 | 1517 | 2a9c5ec2 | 74 | 1,895,400 at step 484 | step 484: reading 0111, code true (1 survivor) | 524 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1481 | 3602 | 41 | B200 |
| m7 | 1763 | c06097a7 | 74 | 1,827,720 at step 470 | step 470: reading 0111, code true (1 survivor) | 503 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 2966 | 32 | B200 |
| m7 | 1763 | 2a9c5ec2 | 74 | 1,827,720 at step 478 | step 478: reading 0111, code true (1 survivor) | 507 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 3488 | 34 | B200 |
| m8 | 1333 | c06097a7 | 74 | 2,963,520 at step 480 | step 480: reading 0111, code true (1 survivor) | - | 0.2% | ood 0.0%, test 0.3% | 0 / 0 | 1500 | 3436 | 83 | B200 |
| m8 | 1333 | 2a9c5ec2 | 74 | 2,963,520 at step 445 | step 445: reading 0111, code true (1 survivor) | 488 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1991 | 19 | B200 |
| m9 | 1927 | 2a9c5ec2 | 74 | 1,833,048 at step 511 | step 511: reading 0111, code true (1 survivor) | 548 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 1912 | 22 | B200 |
| m10 | 1739 | 2a9c5ec2 | 74 | 1,832,220 at step 636 | step 636: reading 0111, code true (1 survivor) | 673 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 1712 | 23 | B200 |
<!-- /TIERS_EARLIER -->

**Dead ends, and how hard each was tried.**

| what | tried on | result | replaced by |
|---|---|---|---|
| the tournament's soft weights (a learned weight per contestant, LR 0.5, pruned 12 nats behind) | tiny core: E1, M2, M8, one seed each | worked, with the same survivors as pruning at once, but after 13 steps of re-scoring every contestant | pruning at the first failed answer, once the machine is exact |
| the writer's code learning under the provisional code | full run: E7 (wave 1) | stuck on 2 of 10 values for 800 steps (probabilities clipped below 10^-6) | no writer-code lesson until the answers choose |
| the reading mixture, with its 10^-6 floor, pulling toward the chosen reading | full runs: M8 (wave 1); forced choices on M7 at steps 200 and 480 | M8 never moved (two pointers to flip, late); the M7 flips (one pointer) worked within 40 steps | the chosen reading's own log-probability, field by field |
| that log-probability pull with the shared pointer shape free | full runs: E2, E6 (wave 2) | the shape switched to "offsets"; untrained writer slots wrote over the answer at evaluation (13%, 12%) | the shape held, and the writer's pointer lesson over every slot |
| the machine's square of x in the multiply's ladder, alone | local CPU runs: E8, E10; every fixed-modulus run of waves 2 and 3 | moved the local E10 and E10 seed 74 past the stall, but 3 of wave 3's 16 runs still stalled (M2, E10 seeds 1 and 2) | the column rules scored whole (the square kept too) |

The census of codes itself, the tie, and pruning at the first failed answer never needed a second try.

### 6. H1 and the other tasks

<!-- H1_NOTES -->
**H1 through its untouched official manifest, on one H100: certified**, the final build byte for byte (`0805b7d3`,
record `v3k_h1_official`). T = 64 on both ladders, every rung 768/768; test 9,999/9,999, ood_t and ood_n_t
10,002/10,002 each; training 1,191 s, evaluation 42 s; frozen at step 645; long division chosen (selector 0.9986)
with the plain square.

**It is v2's run, step for step.** On H1 the reader's facts pin the code at step 165 (`LR digit_codes
pinned_by_facts`), so nothing gated on an ambiguous code ever runs. Against v2's official record (`gen2_h1_official`),
47 of the 48 log lines (timings and v3's own lines left out) are identical: every schedule event, the tables, the
wiring, the selector, the audit's evaluation counts. The 48th is the audit's tally of training visits to the
subtraction table, 0.5% higher: in the first steps, before the reader commits to a reading, it reads every N as the
same number, so the ladder's verdicts (it has no lesson yet) also run on the square of x. The official runs of the
last three v3 builds (`v3f_h1_official` on `32a1d154`, `v3g_h1_official` on `2a9c5ec2`, `v3k_h1_official` on the
final `0805b7d3`) agree in all 48 lines.

**8 seeds** (74, 1–7, the official manifest with the seed changed, on B200s, build `32a1d154`): <!-- SEEDS -->**all 8 certified**: T = 64 on both ladders, every rung 768/768, every split exact, frozen at steps
629 to 749 (v2's CPU seeds: 629 to 750), long division chosen (selector 0.9985 to 0.9993) with the plain square, in 693
to 1,265 s of training.<!-- /SEEDS --> The final build differs from `32a1d154` only inside code that needs an ambiguous code or a
chosen one or a batch whose rows all share one N, which H1 never reaches once its reader has committed, and the
builds' official runs agree line for line.

**The other tasks** (H1's prompts with other answers, and the sampled-moduli tiers, every one with its code pinned by
the facts, so v3 is v2 there; at most 1,500 steps, the tiers with the relaxed budget, B200s):
<!-- OTHER_SUMMARY -->
- **The formulas** (H1's prompts, other answers): rho (y² + 1), ten (10·y²), cube, mont (Montgomery's y²·10⁻⁸),
  and the held-out tenrho (10·y² + 1), plus3 (y² + 3) and xfirst (H1 with its fields reordered): **all certified**, T = 64
  on both ladders, each with the lesson set and round options its formula needs (Montgomery's reduction for mont, the
  square times 10 for ten, + 1, + 3 and times the round's input where the formula has them), frozen at steps 562 to 698
  (builds `32a1d154` and `2a9c5ec2`, which agree with the final build wherever the facts pin the code).
  <!-- MINUS1 -->minus1 (y² − 1, outside the round menu) still fails as in v2: every check exact, no lesson set and round combination fitting a single answer (0.0%, cut by the deadline at step 1,234).<!-- /MINUS1 -->
- **The sampled-moduli tiers** (relaxed budget): E4, M3 and M5 certified T = 64 on both ladders<!-- M4 -->, and so did M4 (every training row at T = 8: v2's fix)<!-- /M4 -->.
  E3 and E5 failed on seed 74 (1.6%, 1.5%) the way v1 failed E3 on seed 74 (LESSONS.md §4): the reader committed to
  its reading only at steps 686 and 764, and the table was not exact within 1,500 steps. v2, run on the same data and
  hardware (`records/diag/v2_e3_s74`, `v2_e5_s74`), gives the same log line for line apart from the audit's visit
  tally. That is the slow seed of the general file, not something v3 changed; LESSONS.md certified E3 on this seed
  with 3,000 steps.
<!-- /OTHER_SUMMARY -->

<!-- H1 -->
| run | task | file | device | score | splits | max T seen / unseen | steps | training s | evaluation s | where it stopped | lesson set chosen (selector) | round options | each set's chain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_h1_official | h1 | c06097a7 | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 1174 | 46 | froze at step 645 | below (lowzero:0.0014,below:0.9986) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3f_h1_official | h1 | 32a1d154 | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 1157 | 42 | froze at step 645 | below (lowzero:0.0014,below:0.9986) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3f_h1_seed1 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 1188 | 44 | froze at step 630 | below (lowzero:0.0007,below:0.9993) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| v3f_h1_seed2 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 951 | 28 | froze at step 643 | below (lowzero:0.0010,below:0.9990) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: trial multiples |
| v3f_h1_seed3 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 1265 | 43 | froze at step 629 | below (lowzero:0.0009,below:0.9991) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3f_h1_seed4 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 877 | 26 | froze at step 749 | below (lowzero:0.0009,below:0.9991) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: trial multiples |
| v3f_h1_seed5 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 701 | 25 | froze at step 646 | below (lowzero:0.0009,below:0.9991) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| v3f_h1_seed6 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 720 | 28 | froze at step 647 | below (lowzero:0.0010,below:0.9990) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3f_h1_seed7 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 939 | 25 | froze at step 720 | below (lowzero:0.0012,below:0.9988) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| v3f_h1_seed74 | h1 | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 693 | 25 | froze at step 640 | below (lowzero:0.0015,below:0.9985) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3g_h1_official | h1 | 2a9c5ec2 | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 1088 | 48 | froze at step 645 | below (lowzero:0.0014,below:0.9986) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3k_h1_official | h1 | v3 | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 1191 | 42 | froze at step 645 | below (lowzero:0.0014,below:0.9986) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
<!-- /H1 -->

<!-- OTHER -->
| run | task | file | device | score | splits | max T seen / unseen | steps | training s | evaluation s | where it stopped | lesson set chosen (selector) | round options | each set's chain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3f_rho_s74 | rho | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 568 | 26 | froze at step 646 | below (lowzero:0.0017,below:0.9983) | rmul=one radd=1 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3f_ten_s74 | ten | 32a1d154 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 564 | 25 | froze at step 641 | below (lowzero:0.0020,below:0.9980) | rmul=ten radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3g_e3_s74 | e3 | 2a9c5ec2 | B200 | 1.6% | ood 2.6%, test 0.5% | 0 / 0 | 1500 | 1103 | 3 | subtraction table never exact (108/200 entries right) | - (no set's wiring lessons joined) | - | - |
| v3g_e4_s74 | e4 | 2a9c5ec2 | B200 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 3132 | 36 | certified without freezing | below (lowzero:0.0002,below:0.9998) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3g_e5_s74 | e5 | 2a9c5ec2 | B200 | 1.5% | ood 2.3%, test 0.8% | 0 / 0 | 1500 | 544 | 2 | subtraction table never exact (103/200 entries right) | - (no set's wiring lessons joined) | - | - |
| v3g_m3_s74 | m3 | 2a9c5ec2 | B200 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1784 | 20 | certified without freezing | below (lowzero:0.0003,below:0.9997) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_cube_s74 | cube | 2a9c5ec2 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 620 | 47 | froze at step 646 | below (lowzero:0.0016,below:0.9984) | rmul=one radd=0 rcube=times_x | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_m4_s74 | m4 | 2a9c5ec2 | B200 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1280 | 3603 | 21 | certified without freezing | below (lowzero:0.0003,below:0.9997) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_m5_s74 | m5 | 2a9c5ec2 | B200 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 3042 | 36 | certified without freezing | below (lowzero:0.0002,below:0.9998) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_minus1_s74 | minus1 | 2a9c5ec2 | B200 | 0.0% | ood_n_t 0.0%, ood_t 0.0%, test 0.1% | 0 / 0 | 1234 | 3601 | 84 | stopped in phase C: hard checks 1.0000, no set-and-round fits (answers 0) | below (lowzero:0.2441,below:0.7559) | rmul=one radd=5 rcube=times_x | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_mont_s74 | mont | 2a9c5ec2 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1322 | 88 | froze at step 698 | lowzero (lowzero:0.9998,below:0.0002) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_plus3_s74 | plus3 | 2a9c5ec2 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 558 | 26 | froze at step 646 | below (lowzero:0.0014,below:0.9986) | rmul=one radd=3 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_tenrho_s74 | tenrho | 2a9c5ec2 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 562 | 26 | froze at step 646 | below (lowzero:0.0014,below:0.9986) | rmul=ten radd=1 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| v3h_xfirst_s74 | xfirst | 2a9c5ec2 | B200 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 573 | 26 | froze at step 562 | below (lowzero:0.0013,below:0.9987) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
<!-- /OTHER -->
<!-- /H1_NOTES -->

### 7. Rules

<!-- RULES -->
v3 keeps every rule status of v2 (LESSONS.md §6) and changes these:

| rule | status | why |
|---|---|---|
| 7. no hard-coded algorithm | gray, as before, with one new designed piece | the tie (§2) fixes the borrow half of the table's meaning to the reader's ranking by construction: "a − b borrows exactly when a < b", with the reader's comparison as the premise. It writes no arithmetic (the comparison is the one the reader's Bradley–Terry lesson and the miner already used, and the table still learns every entry), but it is a designer's statement of what a borrow means. The census's hypothesis space, every bijection of ten digit tokens to ten labels, is designed too (base 10, ten digit tokens) |
| 8. end-to-end, unbroken gradient path | gray, as before, unchanged | no new parameter. The tournament's choice reaches the model as lessons (the reader's ranking, the reading, the selector and the round menu), and the prediction path is v2's |
| 14. no data inspection, augmentation, task-specific solvers | gray, as before | the census and the tournament compute over the batches the evaluator hands the model, inside its loss, as the miner does: they read the batch's tokens and answers, enumerate codes (bookkeeping in Python), and run the model's own machine. Nothing reads the data files or makes up numbers |
| 4. the evaluator owns the loop | gray, as before | the digit-order facts, the contestants, the survivors and the choice are Python-side state of the model, like the miner's streaks |
<!-- /RULES -->

### 8. Times

<!-- TIMES_NOTES -->
The final build on B200s (one container per run; their speeds differed by up to 2x). The reading was chosen at 25 to
56 s, phase C began at 210 to 813 s, the code was chosen at 252 to 1,093 s, and the reader read with it 30 to 50 steps
later. M2 is the slow one: its table and cell took until step 934. None of the datasets whose T always has one digit
freezes (T's direction never settles, as in v2), so they train to the deadline (3,600 s, 1,280 to 1,500 steps), with
nothing left to learn after the choice. M1 and M2, which train on T = 16, settle T's direction and freeze.

What v3 adds costs little:
- The code census is one step: enumerating the 10! codes and filtering them takes a few seconds on the CPU, and the
  first batch's 2,555 to 116,791 machine runs a few seconds more on the GPU. After that the survivor (or E1's two)
  costs nothing.
- The layout census scores the 3,822 column rules once (4 s on a laptop CPU in the tiny test), then the one survivor
  of each kind each step.
- The tie lessons add a soft subtraction per field and top part each step from the reading's choice on, like the
  order facts' own lessons.

Against the official budgets: Easy trains for 60 s, and every Easy run here was still in phase A or B at 60 s. Medium
trains for 600 s on an H100, slower than a B200 by 1.4 to 2.6 times (tools/README.md). Most Medium runs would stop
before the choice. As LESSONS.md did for the tiers, these results use the relaxed budget (3,600 s, at most 1,500
steps).
<!-- /TIMES_NOTES -->

<!-- TIMES -->
| task | seed | reading chosen | digit lessons | phase C | code chosen | reader read with it | frozen | training s |
|---|---|---|---|---|---|---|---|---|
| e1 | 74 | 72 (56 s) | 255 (280 s) | 398 (503 s) | 489 (778 s) | 536 (928 s) | - | 3602 |
| e2 | 74 | 61 (29 s) | 232 (135 s) | 374 (275 s) | 407 (321 s) | 460 (426 s) | - | 3600 |
| e6 | 74 | 76 (34 s) | 393 (280 s) | 532 (436 s) | 560 (479 s) | 605 (556 s) | - | 3602 |
| e7 | 74 | 75 (42 s) | 300 (249 s) | 462 (454 s) | 490 (500 s) | 534 (599 s) | - | 3601 |
| e8 | 74 | 75 (45 s) | 300 (251 s) | 462 (468 s) | 490 (515 s) | 534 (620 s) | - | 3600 |
| e9 | 74 | 65 (44 s) | 233 (196 s) | 376 (375 s) | 410 (434 s) | 462 (557 s) | - | 3602 |
| e10 | 1 | 97 (38 s) | 405 (414 s) | 546 (661 s) | 578 (741 s) | 609 (830 s) | - | 3600 |
| e10 | 2 | 102 (47 s) | 401 (458 s) | 540 (725 s) | 582 (845 s) | 615 (953 s) | - | 3602 |
| e10 | 74 | 97 (44 s) | 410 (485 s) | 549 (738 s) | 624 (955 s) | 656 (1086 s) | - | 3603 |
| m1 | 74 | 78 (27 s) | 192 (96 s) | 331 (245 s) | 361 (302 s) | 411 (417 s) | 645 (1392 s) | 1536 |
| m2 | 1 | 123 (22 s) | 412 (122 s) | 550 (243 s) | 582 (289 s) | 616 (382 s) | 851 (1309 s) | 1430 |
| m2 | 74 | 173 (39 s) | 691 (381 s) | 1002 (813 s) | 1071 (1093 s) | 1102 (1221 s) | 1331 (2322 s) | 2440 |
| m6 | 74 | 72 (32 s) | 207 (149 s) | 351 (303 s) | 383 (343 s) | 428 (423 s) | - | 3570 |
| m7 | 74 | 71 (25 s) | 191 (89 s) | 334 (210 s) | 363 (252 s) | 394 (298 s) | - | 3585 |
| m8 | 74 | 60 (28 s) | 158 (86 s) | 302 (227 s) | 337 (297 s) | 388 (399 s) | - | 3602 |
| m9 | 74 | 78 (25 s) | 232 (125 s) | 373 (266 s) | 404 (309 s) | 447 (406 s) | - | 3603 |
| m10 | 74 | 83 (23 s) | 400 (243 s) | 535 (387 s) | 562 (430 s) | 601 (495 s) | - | 3603 |
<!-- /TIMES -->

### 9. Reproduce

<!-- REPRODUCE -->
v3 is built from v2, which is never modified (each patch must match exactly once; the build checks v2's md5):

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python fixed/make_v3.py
```

The fixed-modulus datasets, generated locally by the official generator into a scratch folder for the tiny tests
(`scripts/generate_datasets.sh`'s 14 `fixed_n` commands, output directories redirected; it needs Python 3.10+ with
torch and jsonargparse). The census and the tiny cores then run on the laptop's CPU, each well under 2 minutes:

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python fixed/code_census.py DATA_DIR [DATA_DIR ...] > fixed/census.jsonl
```

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python fixed/tiny_tie.py fixed=323 tie=1 code=wrong code_seed=0 seed=0
```

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python fixed/tiny_tournament.py DATA_DIR mode=hard
```

Full runs go through the shared runner (`tools/modal_runner.py`, README there), with the plans in `fixed/plans/`:

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode plan --plan general_lessons/fixed/plans/PLAN.json --out general_lessons/fixed/records --cap 5
```

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python fixed/summarize_fixed.py fill fixed/FIXED_MODULUS.md
```

H1's log against v2's official run, line for line (timings and v3's own lines left out):

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python fixed/summarize_fixed.py same records/gen2_h1_official.json fixed/records/v3k_h1_official.json
```
<!-- /REPRODUCE -->

### 10. Files

<!-- FILES -->
New files only; v2 and every certified file are untouched.
- `general_lessons/submission_general_v3.py`: the file (md5 `0805b7d3…`, built by `fixed/make_v3.py` from
  `submission_general_v2.py`);
- `general_lessons/fixed/`:
  - `FIXED_MODULUS.md`: this write-up;
  - `make_v3.py`: the build (every patch an exact text that must occur once in v2);
  - `code_census.py`, `census.jsonl`: §1's exact census;
  - `tiny_tie.py`: §2's tiny table core (from `../tiny_table.py`); `tiny_tournament.py`: §3's tournament on real
    batches with v3's own code; results in `tiny/` (`tie_*`, `tourn_*`), and `tiny/layout/` holds the local E8 and E10
    runs of §5 (the layout's leading rules logged);
  - `summarize_fixed.py`: this write-up's tables, from `records/` and `tiny/`;
  - `variants/`: diagnostics, not submissions: a forced choice of code at a given step (`v3_force_leader.py`,
    `v3_force480_*`), the layout's leaders logged (`v3_layout_diag.py`) and the square in the ladder always on
    (`v3_layout_square.py`);
  - `plans/`, and `records/` (every full run: log, result, manifest, exact source; the build's md5 tells the wave:
    `c06097a7` wave 1, `32a1d154` wave 2, `2a9c5ec2` wave 3, `0805b7d3` the final build). `records/diag/` holds the
    forced flips and the v2 controls on E3 and E5. A launch on build `5e5fe913` (the writer's gate alone) was stopped
    before any record, once wave 1 showed the reading problem.
<!-- /FILES -->
