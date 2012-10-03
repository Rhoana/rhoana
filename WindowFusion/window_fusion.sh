echo "Window Fusion" $*
DIR=$(dirname ${BASH_SOURCE[0]})
ARGS=$(printf ",'%s'" $* | cut -c2-)
export MATLABPATH="${MATLABPATH}:${DIR}"
echo Path is $MATLABPATH
matlab -nojvm -nodesktop -nosplash -r "window_fusion($ARGS)"
if [ -f "${@: -1}" ]
then
    exit 0
fi
exit 1
