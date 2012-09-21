echo "Dicing" $*
DIR=$(dirname ${BASH_SOURCE[0]})
ARGS=$(printf ",'%s'" "$@" | cut -c2-)
MATLABPATH="$DIR" bsub -K -q short_serial matlab -r -nojvm -nodesktop -nosplash -r "dice_block($ARGS); quit"
if [ -f "${@: -1}" ]
then
    exit 0
fi
exit 1

