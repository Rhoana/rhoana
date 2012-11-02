#!/bin/bash
echo "Dicing" $*
DIR=$(dirname ${BASH_SOURCE[0]})
ARGS=$(printf ",'%s'" $* | cut -c2-)
MATLABPATH="$DIR" bsub -K -q short_serial matlab -r -nojvm -nodesktop -nosplash -r "try, dice_block($ARGS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
if [ -f "${@: -1}" ]
then
    exit 0
fi
exit 1

