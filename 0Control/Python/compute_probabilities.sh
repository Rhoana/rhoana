#!/bin/bash
set -e

echo "Compute_Probabilities" $*

if [ -f "$2" ]
then
    exit 0
fi

SLICE_FILE=$1
CUBEDIR=${CONNECTOME}/Pipeline/CubeDicing
COORDS=$(printf ",%s" ${*:3})
if [ $# -eq 2 ]
then
    COORDS=""
fi

MATLABPATH="$CUBEDIR" matlab -nojvm -nodesktop -nosplash -r "try, compute_probabilities('$SLICE_FILE', '${CUBEDIR}/forest_TS1_TS3.mat','$2'$COORDS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
