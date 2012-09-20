echo "Dicing" $*
DIR=$(dirname ${BASH_SOURCE[0]})
# move to 1-based line indexing
IDX=$(expr $2 + 1)
SLICE_FILE=$(sed "${IDX}q;d" $1) 
MATLABPATH="$DIR" bsub -K -q short_serial matlab -r -nojvm -nodesktop -nosplash -r "segment_image('$SLICE_FILE', '${DIR}/forest_TS1_TS3.mat','$3'); quit" > >(tee ~/matlablogs/stdout.$$.txt) 2> >(tee ~/matlablogs/stderr.$$.txt >&2)
if [ -f $3 ]
then
    exit 0
fi
exit 1

