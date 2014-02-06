#!/bin/bash
set -e

echo "Extract Overlay Plane Python" $*
ulimit -c 0
RLDIR="${CONNECTOME}/Relabeling"
python -u "${RLDIR}"/extract_overlay_plane.py $*
