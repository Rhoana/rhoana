#!/bin/bash
set -e

echo "Window Fusion" $*

ARGS=$(printf ",'%s'" $* | cut -c2-)
WFDIR=${CONNECTOME}/Pipeline/WindowFusion

export MATLABPATH="${MATLABPATH}:${WFDIR}"
matlab -nojvm -nodesktop -nosplash -r "try, window_fusion($ARGS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
