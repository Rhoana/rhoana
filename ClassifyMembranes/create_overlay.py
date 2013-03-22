import sys
import os
import h5py
import png
import subprocess
import numpy as np

image = sys.argv[1]
classifier = sys.argv[2]
overlay = sys.argv[3]
mydir = os.path.split(__file__)[0] or '.'

border_classifier = os.path.join(mydir, "border_classifier.tmp")
inout_classifier = os.path.join(mydir, "inout_classifier.tmp")

bout = open(border_classifier, 'w')
iout = open(inout_classifier, 'w')
for l in open(classifier):
    which, rule = l.split(' ', 1)
    if which == 'BORDER':
        bout.write(rule)
    else:
        iout.write(rule)
bout.close()
iout.close()

# run classifiers
print [os.path.join(mydir, 'classify_image'), image, border_classifier,
                       os.path.join(mydir, 'borders.tmp.hdf5')]
subprocess.check_call([os.path.join(mydir, 'classify_image'), image, border_classifier,
                       os.path.join(mydir, 'borders.tmp.hdf5')])
subprocess.check_call([os.path.join(mydir, 'classify_image'), image, inout_classifier,
                       os.path.join(mydir, 'inout.tmp.hdf5')])

# extract classes
border_probs = h5py.File(os.path.join(mydir, 'borders.tmp.hdf5'), 'r')['probabilities'][...]
inout_probs = h5py.File(os.path.join(mydir, 'inout.tmp.hdf5'), 'r')['probabilities'][...]

outim = np.dstack((border_probs, border_probs, border_probs))
# red = 
outim[:, :, 2] = (1 - border_probs) * inout_probs
outim[:, :, 0] = (1 - border_probs) * (1 - inout_probs)
outim = (outim * 255).astype(np.uint8)

f = open(overlay, "w")
writer = png.Writer(outim.shape[1], outim.shape[0])
writer.write(f, outim.reshape(-1, outim.shape[1] * 3))
