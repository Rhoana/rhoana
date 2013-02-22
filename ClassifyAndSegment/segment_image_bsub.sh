#!/bin/bash
echo "Segmenting" $*

cp /n/lichtmanfs1/thouis_jones/Connectome/Pipeline/0Control/Swift/connectome/$3 $3
# short circuit
if [ -f "$3" ]
then
    echo "Copied existing"
    exit 0
fi

DIR=$(dirname ${BASH_SOURCE[0]})
# move to 1-based line indexing
IDX=$(expr $2 + 1)
SLICE_FILE=$(sed "${IDX}q;d" $1) 
MATLABPATH="$DIR" bsub -R "rusage[mem=5000]" -K -q normal_serial matlab -nojvm -nodesktop -nosplash -r "try, segment_image('$SLICE_FILE', '${DIR}/forest_TS1_TS3.mat','$3'), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
if [ -f "$3" ]
then
    exit 0
fi
exit 1
