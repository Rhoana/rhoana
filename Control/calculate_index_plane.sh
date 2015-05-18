#!/bin/bash
set -e

echo "Calculate Index Python" $*
ulimit -c 0
RLDIR="${CONNECTOME}/Relabeling"
python -u "${RLDIR}"/calculate_index_plane_seg_size.py $*
