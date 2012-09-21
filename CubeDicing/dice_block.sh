echo "Dicing" $*
DIR=$(dirname ${BASH_SOURCE[0]})
ARGS=$(printf ",'%s'" "$@" | cut -c2-)
MATLABPATH="$DIR" matlab -nojvm -nodesktop -nosplash -r "dice_block($ARGS); quit"
if [ -f "${@: -1}" ]
then
    exit 0
fi
exit 1

