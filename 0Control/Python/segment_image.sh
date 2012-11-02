#!/bin/bash
echo "Segmenting" $*

if [ -f "$2" ]
then
    exit 0
fi

SLICE_FILE=$1
CUBEDIR=${CONNECTOME}/Pipeline/CubeDicing

MATLABPATH="$CUBEDIR" matlab -nojvm -nodesktop -nosplash -r "try, segment_image('$SLICE_FILE', '${CUBEDIR}/forest_TS1_TS3.mat','$2'), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
