#!/bin/bash
set -e

echo "Clean block" $*

RLDIR=${CONNECTOME}/Cleanup
python "${RLDIR}/clean_block.py" $*
