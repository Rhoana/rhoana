import os
import sys
import subprocess

# fetch LSF job index.  NB: LSF numbers from 1, but we will shift to 0-based
jobidx = int(os.environ['LSB_JOBINDEX']) - 1
assert jobidx >= 0
args = [s.replace('JOBINDEX', str(jobidx)) for s in sys.argv[1:]]

slice_file = open(args[0]).read()  # slice files are indirect
output_file = args[1]
coords = "'" + "','".join(args[2:]) + "'"

if os.path.exists(output_file):
    # Short circuit
    print output_file, "already exists"
    sys.exit(0)

print "Computing probabilities", jobidx, args

cubedir = os.path.join(os.environ['CONNECTOME'], 'Pipeline', 'CubeDicing')
os.environ['MATLABPATH'] = cubedir

command = "try, compute_probabilities('%s', '%s/forest_TS1_TS3.mat', '%s', %s), catch err, display(getReport(err, 'extended')), exit(1), end, exit(0)"
command = command % (slice_file, cubedir, output_file, coords)
print "Running matlab command:", command
subprocess.check_call(['matlab', '-nojvm', '-nodesktop', '-nosplash', '-r', command], env=os.environ) 
