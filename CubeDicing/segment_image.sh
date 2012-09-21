echo "Segmenting" $*

cp /Users/thouis/Connectome/Pipeline/0Control/Swift/connectome/$3 $3
# short circuit
if [ -f "$3" ]
then
    exit 0
fi


DIR=$(dirname ${BASH_SOURCE[0]})
# move to 1-based line indexing
IDX=$(expr $2 + 1)
SLICE_FILE=$(sed "${IDX}q;d" $1) 
XXX MATLABPATH="$DIR" matlab -nojvm -nodesktop -nosplash -r "segment_image('$SLICE_FILE', '${DIR}/forest_TS1_TS3.mat', '$3'); quit"
if [ -f "$3" ]
then
    exit 0
fi
exit 1
