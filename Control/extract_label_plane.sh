#!/bin/bash
set -e

echo "Extract Label Plane New Python" $*
ulimit -c 0
RLDIR="${CONNECTOME}/Relabeling"
python -u "${RLDIR}"/extract_label_plane.py $*
