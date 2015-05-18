#!/bin/bash
set -e

echo "Merge Index" $*

RLDIR=${CONNECTOME}/Relabeling
${VIRTUAL_ENV}/bin/python "${RLDIR}/merge_index.py" $*
