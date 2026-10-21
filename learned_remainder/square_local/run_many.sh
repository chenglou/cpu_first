#!/bin/zsh
# run_many.sh STEPS LESSONS SEED [SEED ...]: tiny.py train runs in parallel (4 threads each, ~100 MB each); one log per run
steps=$1; L=$2; shift 2
for s in "$@"; do
  ~/github/sotaku/venv/bin/python -W ignore tiny.py train $L $s $steps > logs/t_${L}_s${s}.log 2>&1 &
done
wait
for s in "$@"; do echo "$L seed $s: $(grep '^step' logs/t_${L}_s${s}.log | tail -1 | cut -c1-140) | $(grep 'done_at' logs/t_${L}_s${s}.log | sed 's/.*done_at/done_at/')"; done
