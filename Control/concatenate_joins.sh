#!/bin/bash
set -e

echo "Join Concatenation" $*

RLDIR=${CONNECTOME}/Relabeling
${VIRTUAL_ENV}/bin/python "${RLDIR}/concatenate_joins.py" $*
