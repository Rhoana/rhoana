#!/bin/bash
set -e
echo "Pairwise matching" $*

ARGS=$(printf ",'%s'" $* | cut -c2-)
PMDIR=${CONNECTOME}/Pipeline/PairwiseMatching

export MATLABPATH="${MATLABPATH}:${PMDIR}"
matlab -nojvm -nodesktop -nosplash -r "try, pairwise_match_labels($ARGS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
