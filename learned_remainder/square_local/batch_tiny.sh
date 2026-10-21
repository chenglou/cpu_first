#!/bin/zsh
# batch_tiny.sh STEPS "LESSONS:SEED ..." : tiny.py train runs, 4 at a time (each ~100 MB, 4 threads)
steps=$1; shift
jobs_=(${=1})
i=0
for js in $jobs_; do
  L=${js%%:*}; s=${js##*:}
  ~/github/sotaku/venv/bin/python -W ignore tiny.py train $L $s $steps > logs/t_${L}_s${s}.log 2>&1 &
  i=$((i+1))
  if (( i % 4 == 0 )); then wait; fi
done
wait
for js in $jobs_; do
  L=${js%%:*}; s=${js##*:}
  echo "$L seed $s: $(grep '^step' logs/t_${L}_s${s}.log | tail -1 | cut -c1-130) | $(grep 'done_at' logs/t_${L}_s${s}.log | sed 's/.*done_at/done_at/')"
done
