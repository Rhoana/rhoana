echo "Dicing" $*
DIR=$(dirname ${BASH_SOURCE[0]})
SLICE_FILE=$(sed "${2}q;d" $1) 
MATLABPATH="$DIR" matlab -r -nojvm -nodesktop -nosplash -r "segment_image('$SLICE_FILE', '${DIR}/forest_TS1_TS3.mat','$3'); quit" > >(tee ~/matlablogs/stdout.$$.txt) 2> >(tee ~/matlablogs/stderr.$$.txt >&2)
echo "Exit status" $?
