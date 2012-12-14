#!/bin/bash
set -e

echo "Window Fusion New Python" $*
ulimit -c 0

# ARGS=$(printf ",'%s'" $* | cut -c2-)
WFDIR="${CONNECTOME}/Pipeline/WindowFusion"

python -u "${WFDIR}"/window_fusion_cpx.py $*
# export MATLABPATH="${MATLABPATH}:${WFDIR}"
# matlab -nojvm -nodesktop -nosplash -r "try, window_fusion($ARGS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
