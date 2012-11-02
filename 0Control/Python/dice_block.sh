#!/bin/bash
set -e
echo "Dicing" $*

ARGS=$(printf ",'%s'" $* | cut -c2-)
CUBEDIR=${CONNECTOME}/Pipeline/CubeDicing

MATLABPATH="$CUBEDIR" matlab -r -nojvm -nodesktop -nosplash -r "try, dice_block($ARGS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"

