#!/bin/bash
set -e

echo "Remap block" $*

RLDIR=${CONNECTOME}/Pipeline/Relabeling
${VIRTUAL_ENV}/bin/python "${RLDIR}/remap_block.py" $*
