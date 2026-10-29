# The blind reviews

Four Claude agents each reviewed one file, with the same brief. Files were renamed A–D, so no reviewer knew whose file
it had, and none was told what the organizers concluded. The reports here are condensed in our words; line numbers
refer to the files as reviewed.

| File | Upload |
|---|---|
| A | Sahil Verma, `9fa00118` (rank 3) |
| B | the deep file, `integration/legal_deep/submission_stage3_deep.py` (`56acca73`) |
| C | CodeReclaimers, `58311e4a`: the 29 August version the organizers reviewed (the ranked one is `d83efe07`) |
| D | Yash Kant, `86e9acf2` (rank 2): the organizers' winner |

**The brief, in short.** The reviewers got:
- the contest's rules 1–15;
- the task, x^(2^T) mod N;
- Hard's hidden twist: N's digits reversed in the prompt;
- the certification ladder.

Then they were asked the judges' central question: how much of the number representation, the per-step transition
(squaring, then reduction mod N) and the control across T is learned from the supplied labels? The alternative is that
it's supplied by the author, or fit to targets the code makes itself. Each report gives:
1. a mechanism summary;
2. a component table;
3. findings on rules 7, 8, 9 and 12/14;
4. one of four classes: (a) hard-codes the target algorithm, (b) uses it to assist learning, (c) another rule break,
   (d) learned and clear, plus a 1–10 score for how much is learned;
5. the strongest argument each way.

Reviewers judged from the code alone: no write-ups, and no running the files.
