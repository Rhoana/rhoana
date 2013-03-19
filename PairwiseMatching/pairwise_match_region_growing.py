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
bl1f = h5py.File(block1_path, 'r')
block1 = bl1f['labels']
probs1 = bl1f['probabilities']
bl2f = h5py.File(block2_path, 'r')
block2 = bl2f['labels']
probs2 = bl2f['probabilities']
assert block1.size == block2.size

# extract overlap
lo_block1 = [0, 0, 0];
hi_block1 = [None, None, None]
lo_block2 = [0, 0, 0];
hi_block2 = [None, None, None]

# Adjust overlapping region boundaries for direction
lo_block1[direction] = - 2 * halo_size
hi_block2[direction] = 2 * halo_size;

block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
overlap1 = block1[block1_slice]
overlap2 = block2[block2_slice]

counter = fast64counter.ValueCountPair64()
counter.add_values(overlap1.ravel(), overlap2.ravel())
overlap_labels1, overlap_labels2, overlap_areas = counter.get_counts()

areacounter = fast64counter.ValueCountInt64()
areacounter.add_values(overlap1.ravel())
areacounter.add_values(overlap2.ravel())
areas = dict(zip(*areacounter.get_counts()))

to_merge = []
for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):
    if l1 == 0 or l2 == 0:
        continue
    if inverse[l1] == inverse[l2]:
        continue
    if ((overlap_area > auto_join_pixels) or
        ((overlap_area > minoverlap_pixels) and
         ((overlap_area > minoverlap_single_ratio * areas[l1]) or
          (overlap_area > minoverlap_single_ratio * areas[l2]) or
          ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
           (overlap_area > minoverlap_dual_ratio * areas[l2]))))):
        to_merge.append((inverse[l1], inverse[l2]))
# Merges are handled later

# After merging, we extract half of the overlap, remove all but its outer
# boundary, and run watershed on the probability map to fill in the interior.
probabilities = probs1[block1_slice]
assert np.all(probabilities == probs2[block2_slice])
hi_block1[direction] = - halo_size
lo_block2[direction] = halo_size;
border_block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
border_block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
border_only = np.concatenate((block1[border_block1_slice], block2[border_block2_slice]), direction)
border_only[1:-1, 1:-1, 1:-1] = 0
probabilities = (probabilities * 255).astype(np.uint8)
seeded_region_growing.inplace_region_growing(probabilities, border_only)

# Remap and merge
out1 = h5py.File(outblock1_path + '_partial', 'w')
out2 = h5py.File(outblock2_path + '_partial', 'w')
outblock1 = out1.create_dataset('/labels', block1.shape, block1.dtype, chunks=block1.chunks)
outblock2 = out2.create_dataset('/labels', block2.shape, block2.dtype, chunks=block2.chunks)
outblock1[...] = block1[...]
outblock2[...] = block2[...]

# write the grown regions back into the blocks
outblock1[block1_slice] = border_only
outblock2[block2_slice] = border_only

# copy the probabilities forward
outprobs1 = out1.create_dataset('/probabilities', block1.shape, probs1.dtype, chunks=probs1.dtype)
outprobs2 = out2.create_dataset('/probabilities', block2.shape, probs2.dtype, chunks=probs2.dtype)
outprobs1[...] = probs1[...]
outprobs2[...] = probs2[...]

# copy any previous merge tables to the new output
if 'merges' in bl1f:
    merges = bl1f['merges']
    if len(to_merge):
        merges = np.vstack((merges, to_merge))
else:
    merges = np.array(to_merge).astype(np.uint64)
if merges.size > 0:
    out1.create_dataset('/merges', merges.shape, merges.dtype)[...] = merges

if 'merges' in bl2f:
    merges = bl2f['merges']
    out2.create_dataset('/merges', merges.shape, merges.dtype)[...] = merges

# move to final location
out1.close()
out2.close()

if os.path.exists(outblock1_path):
        os.unlink(outblock1_path)
if os.path.exists(outblock2_path):
        os.unlink(outblock2_path)

os.rename(outblock1_path + '_partial', outblock1_path)
os.rename(outblock2_path + '_partial', outblock2_path)
print "Successfully wrote", outblock1_path, 'and', outblock2_path
