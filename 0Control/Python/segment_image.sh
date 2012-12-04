#!/bin/bash
set -e

echo "Segmenting" $*

if [ -f "$2" ]
then
    exit 0
fi

PROB_FILE=$1
CUBEDIR=${CONNECTOME}/Pipeline/CubeDicing
COORDS=$(printf ",%s" ${*:3})
if [ $# -eq 2 ]
    COORDS=""
fi

MATLABPATH="$CUBEDIR" matlab -nojvm -nodesktop -nosplash -r "try, segment_image('$SLICE_FILE', '$2'$COORDS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
