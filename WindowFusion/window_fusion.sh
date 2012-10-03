echo "Window Fusion" $*
DIR=$(dirname ${BASH_SOURCE[0]})
ARGS=$(printf ",'%s'" $* | cut -c2-)
MATLABPATH="/n/sw/cplex-12.3/cplex/matlab:$DIR" matlab -nojvm -nodesktop -nosplash -r "window_fusion($ARGS)"
if [ -f "${@: -1}" ]
then
    exit 0
fi
exit 1
