#!/bin/bash
set -e

echo "Create Global Map" $*

RLDIR=${CONNECTOME}/Relabeling
python "${RLDIR}/create_global_map.py" $*
