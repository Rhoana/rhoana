DIR=$(dirname ${BASH_SOURCE[0]})
SLICE_FILE=$(sed "${2}q;d" $1) 
MATLABPATH="$DIR" /Applications/MATLAB_R2012a.app/bin/matlab -r -nodesktop -nosplash -r "segment_image('$SLICE_FILE', '${DIR}/forest_TS1_TS3.mat','$3'); quit"
