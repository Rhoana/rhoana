import os
import sys
import glob
import h5py
import mahotas
import numpy as np

input_folder = sys.argv[1]
output_h5 = sys.argv[2]

rot = 0
if len(sys.argv) > 3:
    rot = int(sys.argv[3])

def alphanum_key(s):
    digits = ''.join([c for c in s if c.isdigit()])
    if len(digits) > 0:
        return int(digits)
    return s

def sort_numerically(l):
    l.sort(key=alphanum_key)

input_files = glob.glob(os.path.join(input_folder, '*'))
sort_numerically(input_files)

stack = None

for i, file_name in enumerate(input_files):

    if file_name.endswith('h5') or file_name.endswith('hdf5'):
        infile = h5py.File(file_name)
        im = infile['/probabilities'][...]
    else:
        im = mahotas.imread(file_name)

    if rot != 0:
        im = np.rot90(im, rot)

    if stack is None:
        stack = np.zeros((len(input_files), im.shape[0], im.shape[1]), dtype=im.dtype)
        print 'Generating stack size={0}, dtype={1}.'.format(stack.shape, stack.dtype)
    stack[i,:,:] = im

    print file_name

if os.path.exists(output_h5):
    os.unlink(output_h5)

h5file = h5py.File(output_h5, 'w')

stack_dataset = h5file.create_dataset('stack',
    stack.shape,
    data=stack,
    dtype=stack.dtype,
    chunks=(64, 64, 4),
    compression='gzip')

h5file.close()

print 'Wrote file {0}.'.format(output_h5)
