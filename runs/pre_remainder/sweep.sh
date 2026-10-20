#!/bin/zsh
# usage: sweep.sh <commands-file> <parallel>   (each line: <logname> <args to microlab.ext_preremainder>)
PY=/Users/chenglou/github/one-layer/one-layer-deeper/.venv/bin/python
LIST=${1:A}; cd "${0:A:h}/../.."
grep -v "^#" "$LIST" | grep -v '^$' | xargs -P "$2" -L 1 zsh -c 'name=$1; shift; nice -n 5 '"$PY"' -m microlab.ext_preremainder "$@" > runs/pre_remainder/$name.log 2>&1; echo "done $name: $(tail -1 runs/pre_remainder/$name.log | cut -c1-200)"' _
