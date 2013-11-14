#!/bin/bash
set -e

echo "Remap block" $*

RLDIR=${CONNECTOME}/Relabeling
python "${RLDIR}/remap_block.py" $*
