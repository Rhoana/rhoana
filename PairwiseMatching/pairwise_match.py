import sys
from collections import defaultdict
import numpy as np
import os

import h5py
import fast64counter


Debug = False

block1_path, block2_path, direction, halo_size, outblock1_path, outblock2_path = sys.argv[1:]
direction = int(direction)
halo_size = int(halo_size)

###############################
# Note: direction indicates the relative position of the blocks (1, 2, 3 =>
# adjacent in X, Y, Z).  Block1 is always closer to the 0,0,0 corner of the
# volume.
###############################

###############################
# Note that we are still in matlab hdf5 coordinates, so everything is stored ZYX
###############################

###############################
#Change joining thresholds here
###############################
#Join 1 (less joining)
auto_join_pixels = 20000; # Join anything above this many pixels overlap
minoverlap_pixels = 2000; # Consider joining all pairs over this many pixels overlap
minoverlap_dual_ratio = 0.7; # If both overlaps are above this then join
minoverlap_single_ratio = 0.9; # If either overlap is above this then join

# Join 2 (more joining)
# auto_join_pixels = 10000; # Join anything above this many pixels overlap
# minoverlap_pixels = 1000; # Consider joining all pairs over this many pixels overlap
# minoverlap_dual_ratio = 0.5; # If both overlaps are above this then join
# minoverlap_single_ratio = 0.8; # If either overlap is above this then join


print 'Running pairwise matching', " ".join(sys.argv[1:])

# Extract overlapping regions
bl1f = h5py.File(block1_path)
block1 = bl1f['labels']
block2 = h5py.File(block2_path)['labels']
assert block1.size == block2.size

lo_block1 = [0, 0, 0];
hi_block1 = [None, None, None]
lo_block2 = [0, 0, 0];
hi_block2 = [None, None, None]

# Adjust for Matlab HDF5 storage order
direction = 3 - direction

# Adjust overlapping region boundaries for direction
lo_block1[direction] = - 2 * halo_size
hi_block2[direction] = 2 * halo_size;

# extract overlap
block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
block1_overlap = block1[block1_slice]
block2_overlap = block2[block2_slice]

if Debug:
    import pylab
    pylab.imshow(block1[0, :, :])
    pylab.title('block1')
    pylab.figure()
    pylab.imshow(block1_overlap[0, :, :])
    pylab.title('block1 over')
    pylab.figure()
    pylab.imshow(block2[0, :, :])
    pylab.title('block2')
    pylab.figure()
    pylab.imshow(block2_overlap[0, :, :])
    pylab.title('block2 over')
    pylab.figure()

    pylab.show()


# append the blocks, and pack them so we can use the fast 64-bit counter
stacked = np.vstack((block1_overlap, block2_overlap))
inverse, packed = np.unique(stacked, return_inverse=True)
packed = packed.reshape(stacked.shape)
packed1 = packed[:block1_overlap.shape[0], :, :]
packed2 = packed[block1_overlap.shape[0]:, :, :]

counter = fast64counter.ValueCountInt64()
counter.add_values_pair32(packed1.astype(np.int32).ravel(), packed2.astype(np.int32).ravel())
overlap_labels1, overlap_labels2, overlap_areas = counter.get_counts_pair32()

areacounter = fast64counter.ValueCountInt64()
areacounter.add_values(packed.ravel())
areas = dict(zip(*areacounter.get_counts()))

to_merge = []
to_steal = []
for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):
    if l1 == 0 or l2 == 0:
        continue
    if ((overlap_area > auto_join_pixels) or
        ((overlap_area > minoverlap_pixels) and
         ((overlap_area > minoverlap_single_ratio * areas[l1]) or
          (overlap_area > minoverlap_single_ratio * areas[l2]) or
          ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
           (overlap_area > minoverlap_dual_ratio * areas[l2]))))):
        to_merge.append((l1, l2))
    else:
        to_steal.append((overlap_area, l1, l2))

# Merges are handled later

if Debug:
    import pylab
    pylab.figure()
    pylab.imshow(packed1[0, :, :] % 13)
    pylab.title('packed1before')
    pylab.figure()
    pylab.imshow(packed2[0, :, :] % 13)
    pylab.title('packed2before')


# Process steals
packed1_face = packed1[tuple(0 if i == direction else slice(None, None) for i in range (3))]
packed2_face = packed2[tuple(-1 if i == direction else slice(None, None) for i in range (3))]

faceareacounter = fast64counter.ValueCountInt64()
faceareacounter.add_values(packed1_face.ravel())
faceareacounter.add_values(packed2_face.ravel())
face_areas = defaultdict(int)
face_areas.update(dict(zip(*faceareacounter.get_counts())))
for _, l1, l2 in reversed(sorted(to_steal)):  # work largest to smallest
    if face_areas[l1] >= face_areas[l2]:
        packed2[(packed1 == l1)] = l1
    else:
        packed1[(packed2 == l2)] = l2

if Debug:
    import pylab
    pylab.figure()
    pylab.imshow(packed1[0, :, :] % 13)
    pylab.title('packed1')
    pylab.figure()
    pylab.imshow(packed2[0, :, :] % 13)
    pylab.title('packed2')
    pylab.show()


# Remap and merge
block1[block1_slice] = inverse[packed1]
block2[block2_slice] = inverse[packed2]

if Debug:
    import pylab
    pylab.imshow(block1[0, :, :] % 13)
    pylab.title('block1')
    pylab.figure()
    pylab.imshow(block2[0, :, :] % 13)
    pylab.title('block2')
    pylab.show()


if Debug:
    # this is slow
    block1data = block1[...]
    block2data = block2[...]
    for l1, l2 in to_merge:
        block1data[block1data == inverse[l1]] = inverse[l2]
        block2data[block2data == inverse[l1]] = inverse[l2]

    import pylab
    pylab.imshow(block1data[0, :, :] % 13)
    pylab.title('block1')
    pylab.figure()
    pylab.imshow(block2data[0, :, :] % 13)
    pylab.title('block2')
    pylab.show()

out1 = h5py.File(outblock1_path + '_partial', 'w')
out2 = h5py.File(outblock2_path + '_partial', 'w')

outblock1 = out1.create_dataset('/labels', block1.shape, block1.dtype, chunks=block1.chunks)
outblock2 = out2.create_dataset('/labels', block2.shape, block2.dtype, chunks=block2.chunks)
outblock1[...] = block1[...]
outblock2[...] = block2[...]

to_merge = np.array(to_merge).reshape((-1, 2))
if 'joins' in bl1f:
    joins = np.vstack((bl1f['joins'][...], to_merge))
else:
    joins = to_merge
if joins.size > 0:
    j = out1.create_dataset('/joins', joins.shape, np.int64)
    j[...] = joins

# move to final location
if os.path.exists(outblock1_path):
        os.unlink(outblock1_path)
if os.path.exists(outblock2_path):
        os.unlink(outblock2_path)

out1.close()
out2.close()
os.rename(outblock1_path + '_partial', outblock1_path)
os.rename(outblock2_path + '_partial', outblock2_path)
print "Successfully wrote", outblock1_path, 'and', outblock2_path
