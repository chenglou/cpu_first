#!/bin/bash
# A memory cap for a local run (ROWCHECK.md §4): runs the command, checks its resident memory (the process and its
# children) every 5 seconds, and kills it when that passes CAP_GB. Exits with the command's code, or 137 when killed.
#
#   bar3/scale/rowcheck/memcap.sh CAP_GB COMMAND ARGS...
cap_kb=$(( $1 * 1024 * 1024 ))
shift
"$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  kids=$(pgrep -P "$pid" | tr '\n' ',')
  rss=$(ps -o rss= -p "${kids}${pid}" 2>/dev/null | awk '{s += $1} END {print s + 0}')
  if [ "$rss" -gt "$cap_kb" ]; then
    echo "memcap: ${rss} KB is over the cap of ${cap_kb} KB; killing $pid" >&2
    pkill -P "$pid" 2>/dev/null
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit 137
  fi
  sleep 5
done
wait "$pid"
