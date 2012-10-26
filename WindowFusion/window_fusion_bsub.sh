echo "Window Fusion bsub" $*
DIR=$(dirname ${BASH_SOURCE[0]})
ARGS=$(printf ",'%s'" $* | cut -c2-)
export MATLABPATH="${MATLABPATH}:${DIR}"
bsub -K -q short_serial matlab -nojvm -nodesktop -nosplash -r "try, window_fusion($ARGS), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
if [ -f "${@: -1}" ]
then
    exit 0
fi
exit 1
