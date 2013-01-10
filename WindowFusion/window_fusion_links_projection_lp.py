import os
import sys
import time
import gc
import operator
from collections import defaultdict

import numpy as np
from scipy.ndimage.measurements import label
import h5py

import fast64counter

##################################################
# Parameters
##################################################
size_compensation_factor = 0.9
chunksize = 64

# NB - both these functions should accept array arguments
# weights for segments
def segment_worth(area):
    return area ** size_compensation_factor
# weights for links
def link_worth(area1, area2, area_overlap):
    min_area = np.minimum(area1, area2)
    max_fraction = area_overlap / np.maximum(area1, area2)
    return max_fraction * (min_area ** size_compensation_factor) + (segment_worth(area1) + segment_worth(area2)) / 2


class timed(object):
    def __init__(self, f):
        self.f = f
        self.total_time = 0.0

    def __call__(self, *args, **kwargs):
        start = time.time()
        val = self.f(*args, **kwargs)
        self.total_time += time.time() - start
        return val

@timed
def unique_labels(depth, seg, values, offset):
    labels = label(values, output=np.int64)[0]
    return labels + (labels > 0) * offset

def count_overlaps(depth, numsegs, labels):
    st = time.time()

    htable = fast64counter.ValueCountInt64()
    # Count areas of each label
    for D in range(depth):
        for Seg in range(numsegs):
            lbls = labels[Seg, D, :, :][...]
            htable.add_values(lbls.ravel())
    keys, areas = htable.get_counts()
    areas = areas[np.argsort(keys)]
    areas[0] = 0

    # sanity check
    assert np.all(np.sort(keys) == np.arange(len(keys)))

    def exclusions():
        for D in range(depth):
            print "exl depth", D
            excls = set()
            for xpos in range(0, labels.shape[2], chunksize):
                for ypos in range(0, labels.shape[3], chunksize):
                    subimages = [labels[Seg, D, xpos:(xpos + chunksize), ypos:(ypos + chunksize)][...].ravel() for Seg in range(numsegs)]
                    excls.update(set(zip(*subimages)))
            for excl in excls:
                yield excl

    def overlaps():
        overlap_areas = fast64counter.ValueCountInt64()
        for D in range(depth - 1):
            print "depth", D
            for xpos in range(0, labels.shape[2], chunksize):
                for ypos in range(0, labels.shape[3], chunksize):
                    subimages_d1 = [labels[Seg, D, xpos:(xpos + chunksize), ypos:(ypos + chunksize)][...].ravel().astype(np.int32) for Seg in range(numsegs)]
                    subimages_d2 = [labels[Seg, D + 1, xpos:(xpos + chunksize), ypos:(ypos + chunksize)][...].ravel().astype(np.int32) for Seg in range(numsegs)]
                    for s1 in subimages_d1:
                        for s2 in subimages_d2:
                            overlap_areas.add_values_32(s1, s2)
        combined_idxs, overlap_areas = overlap_areas.get_counts()
        idxs1 = combined_idxs >> 32
        idxs2 = combined_idxs & 0xffffffff
        mask = (idxs1 > 0) & (idxs2 > 0)
        idxs1 = idxs1[mask]
        idxs2 = idxs2[mask]
        overlap_areas = overlap_areas[mask]
        print len(idxs1), "Overlaps"
        for idx1, idx2, overlap_area in zip(idxs1, idxs2, overlap_areas):
            yield idx1, idx2, link_worth(float(areas[idx1]), float(areas[idx2]), float(overlap_area))

    print "Area counting took", int(time.time() - st), "seconds"

    return areas, exclusions(), overlaps()

def build_model(lpfile, areas, exclusions, overlaps):
    ##################################################
    # Generate the LP problem
    ##################################################
    print "Building MILP problem:"

    st = time.time()

    segments = np.arange(len(areas))

    # Build the LP
    num_segments = len(segments)
    print "  segments", num_segments
    obj_terms = []
    linknames = []

    print "building segment to link map"
    # add links and link constraints
    uplinksets = defaultdict(list)
    downlinksets = defaultdict(list)
    for linkidx, (idx1, idx2, weight) in enumerate(overlaps):
        obj_terms += ["%f link_%d_%d" % (weight, idx1, idx2)]
        linknames += ["link_%d_%d" % (idx1, idx2)]
        l = (idx1, idx2)
        uplinksets[idx1].append(l)
        downlinksets[idx2].append(l)

    print "Adding exclusions"
    def write_exclusions(f):
        for s in range(1, num_segments):
            # activators
            if uplinksets[s]:
                f.write("seg_active_%d - %s >= 0\n" % (s, "-".join(["link_%d_%d" % (l1, l2) for (l1, l2) in uplinksets[s]])))
            if downlinksets[s]:
                f.write("seg_active_%d - %s >= 0\n" % (s, "-".join(["link_%d_%d" % (l1, l2) for (l1, l2) in downlinksets[s]])))
        for excl in exclusions:
            f.write("%s <= 1\n" % "+".join(["seg_active_%d" % s for s in excl if s > 0]))

    lpfile.write("Maximize\n")
    lpfile.write(" + ".join(obj_terms))
    lpfile.write("\n")
    lpfile.write("Subject To\n")
    write_exclusions(lpfile)
    lpfile.write("Binary\n")
    lpfile.write("\n".join(linknames))
    lpfile.write("\n")
    lpfile.write("\n".join(["seg_active_%d" % idx for idx in range(1, num_segments)]))
    lpfile.write("\n")
    lpfile.write("End\n")

if __name__ == '__main__':
    segmentations = h5py.File(sys.argv[1])['cubesegs']

    ##################################################
    # compute all overlaps between multisegmentations
    ##################################################
    numsegs, depth, width, height = segmentations.shape
    print segmentations.shape

    # ensure we can store all the labels we need to
    assert (width * height * depth * numsegs) < (2 ** 31 - 1), \
        "Cube too large.  Must be smaller than 2**31 - 1 voxels."

    largest_index = depth * numsegs * width * height

    st = time.time()

    # Precompute labels, store in HDF5
    block_offset = int(sys.argv[2]) << 32
    output_path = sys.argv[3]

    try:
        lf = h5py.File(output_path, 'r')
        if 'labels' in lf.keys():
            print "Output already generated"
            lf.close()
            sys.exit(0)
    except Exception, e:
        pass

    lf = h5py.File(output_path + '_partial', 'w')
    chunking = [1, 1, chunksize, chunksize]
    # We have to ues int64 for the pandas hash table implementation
    labels = lf.create_dataset('seglabels', segmentations.shape, dtype=np.int64, chunks=tuple(chunking), compression='gzip')
    offset = 0
    for D in range(depth):
        for Seg in range(numsegs):
            temp = unique_labels(D, Seg, segmentations[Seg, D, :, :][...], offset)
            labels[Seg, D, :, :] = temp
            assert temp.max() > offset
            offset = temp.max()
            assert offset < 2**31
    print "Labeling took", int(time.time() - st), "seconds"

    areas, exclusions, overlaps = count_overlaps(depth, numsegs, labels)
    num_segments = len(areas)
    print num_segments, offset
    assert num_segments == offset + 1  # areas includes an area for 0

    st = time.time()
    build_model(open("theproblem_joined_excls.lp", "w"), areas, exclusions, overlaps)
    print "Building MILP took", int(time.time() - st), "seconds"
