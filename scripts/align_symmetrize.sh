#!/bin/bash

# Usage:
#  align_symmetrize source.txt target.txt output.moses method [efmaral options]
# Where method is one of the symmetrization methods from atools (the -c
# argument).

if [ -z $4 ] ; then
    echo "Error: symmetrization argument missing!"
    echo "You might want to try grow-diag-final-and"
    exit 1
fi
SYMMETRIZATION=$4

#SYMMETRIZATION=grow-diag-final-and
#if [ ! -z $4 ] ; then
#    SYMMETRIZATION=$4
#fi

FWD=`mktemp`
BWD=`mktemp`
python3 efmaral.py --verbose -i "$1" "$2" "${@:5}" >"$FWD" &
python3 efmaral.py --verbose -r -i "$1" "$2" "${@:5}" >"$BWD" &
wait
atools -c $SYMMETRIZATION -i "$FWD" -j "$BWD" >"$3"
rm "$FWD" "$BWD"

