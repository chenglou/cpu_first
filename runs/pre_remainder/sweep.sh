#!/bin/zsh
# usage: sweep.sh <commands-file> <parallel>   (each line: <logname> <args to microlab.ext_preremainder>)
PY=/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/f90ac968-2b9c-4f1c-a51e-c2cf0582042a/scratchpad/one-layer-deeper/.venv/bin/python
LIST=${1:A}; cd "${0:A:h}/../.."
grep -v "^#" "$LIST" | grep -v '^$' | xargs -P "$2" -L 1 zsh -c 'name=$1; shift; nice -n 5 '"$PY"' -m microlab.ext_preremainder "$@" > runs/pre_remainder/$name.log 2>&1; echo "done $name: $(tail -1 runs/pre_remainder/$name.log | cut -c1-200)"' _
