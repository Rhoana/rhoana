#!/bin/bash
echo "Segmenting" $*

if [ -f "$2" ]
then
    exit 0
fi

SLICE_FILE=$1
CUBEDIR=${CONNECTOME}/Pipeline/CuebDicing

MATLABPATH="$CUBEDIR" bsub -R "rusage[mem=5000]" -K -q normal_serial matlab -nojvm -nodesktop -nosplash -r "try, segment_image('$SLICE_FILE', '${CUBEDIR}/forest_TS1_TS3.mat','$3'), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
if [ -f "$3" ]
then
    exit 0
fi
exit 1
