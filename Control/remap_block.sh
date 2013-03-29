#!/bin/bash
set -e

echo "Remap block" $*

RLDIR=${CONNECTOME}/Relabeling
${VIRTUAL_ENV}/bin/python "${RLDIR}/remap_block.py" $*
