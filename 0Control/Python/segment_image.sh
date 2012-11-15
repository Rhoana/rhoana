#!/bin/bash
set -e

echo "Segmenting" $*

IMAGE_FILE=$1
PROBABILITY_FILE=$2
OUTPUT=$3

if [ -f "$OUTPUT" ]
then
    exit 0
fi

CUBEDIR=${CONNECTOME}/Pipeline/CubeDicing

MATLABPATH="$CUBEDIR" matlab -nojvm -nodesktop -nosplash -r "try, segment_image('$IMAGE_FILE', '$PROBABILITY_FILE', '$OUTPUT'), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
