#!/bin/bash
set -e

echo "Create Global Map" $*

RLDIR=${CONNECTOME}/Relabeling
${VIRTUAL_ENV}/bin/python "${RLDIR}/create_global_map.py" $*
