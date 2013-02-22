#!/bin/bash
set -e

echo "Segmenting" $*

ORIG_FILE="$1"
PROB_FILE="$2"
OUT_FILE="$3"

if [ -f "$OUT_FILE" ]
then
    echo "short circuit" $OUT_FILE
    exit 0
fi

CUBEDIR=${CONNECTOME}/Pipeline/CubeDicing
COORDS=$(printf ",%s" ${*:4})
if [ $# -eq 3 ]
then
    COORDS=""
fi

echo running "try, segment_image('$ORIG_FILE', '$PROB_FILE', '$OUT_FILE'$COORDS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)", MATLAB_PATH, "$CUBEDIR"
MATLABPATH="$CUBEDIR" matlab -nojvm -nodesktop -nosplash -r "try, segment_image('$ORIG_FILE', '$PROB_FILE', '$OUT_FILE'$COORDS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
echo done

