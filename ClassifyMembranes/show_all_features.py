import sys
import h5py
import pylab


f = h5py.File(sys.argv[1], 'r')
for k in f.keys():
    pylab.figure()
    pylab.imshow(f[k][...], pylab.gray())
    pylab.colorbar()
    pylab.title(k)

localhists = [f[k][...] for k in f.keys() if 'local_hist' in k]
sl = sum(localhists)
print "sum of local hists min", sl.min(), "max", sl.max()

pylab.show()
