import os
import sys
import subprocess

# fetch LSF job index.  NB: LSF numbers from 1, but we will shift to 0-based
jobidx = int(os.environ['LSB_JOBINDEX']) - 1
assert jobidx >= 0
args = [s.replace('JOBINDEX', str(jobidx)) for s in sys.argv[1:]]

reassembler = os.path.join(os.environ['CONNECTOME'], 'Pipeline', 'CubeDicing', 'reassemble.py')
subprocess.check_call(['python', reassembler] + args)
