#!/bin/bash
set -e

CUBEDIR=${CONNECTOME}/Pipeline/CubeDicing

# Hand off to python
python ${CUBEDIR}/reassemble.py "${@}"
