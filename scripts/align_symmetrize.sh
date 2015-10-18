#!/bin/bash

# Usage: align_symmetrize source.txt target.txt output.moses [method]
# Where method is one of the symmetrization methods from atools (the -c
# argument).

SYMMETRIZATION=grow-diag-final-and
if [ ! -z $4 ] ; then
    SYMMETRIZATION=$4
fi
python3 efmaral.py -i "$1" "$2" >align.forward &
python3 efmaral.py -r -i "$1" "$2" >align.back &
wait
atools -c $SYMMETRIZATION -i align.forward -j align.back >"$3"
rm align.forward align.back

