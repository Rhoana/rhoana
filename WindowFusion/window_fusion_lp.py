import os
import sys
import time
import gc
import operator

import numpy as np
from scipy.ndimage.measurements import label
import h5py
import pandas

##################################################
# Parameters
##################################################
size_compensation_factor = 0.9

# NB - both these functions should accept array arguments
# weights for segments
def segment_worth(area):
    return area ** size_compensation_factor
# weights for links
def link_worth(area1, area2, area_overlap):
    min_area = np.minimum(area1, area2)
    max_fraction = area_overlap / np.maximum(area1, area2)
    return max_fraction * (min_area ** size_compensation_factor)


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
    areas = {}

    st = time.time()

    # Count areas of each label
    for D in range(depth):
        for Seg in range(numsegs):
            lbls = labels[Seg, D, :, :][...]
            keys, counts = pandas.lib.value_count_int64(lbls.ravel())
            areas.update(zip(keys, counts))
    areas[0] = 0

    def exclusions():
        for D in range(depth):
            print "depth", D
            for Seg in range(numsegs):
                labels1 = labels[Seg, D, :, :][...]
                labels1 <<= 32

                l1flat = labels1.flat
                # all overlaps at this same depth
                for Seg2 in range(Seg + 1, numsegs):
                    labels2 = labels[Seg2, D, :, :][...]
                    np.add(labels1, labels2, labels2)
                    keys, counts = pandas.lib.value_count_int64(labels2.ravel())
                    for k in keys:
                        idx1 = int(k >> 32)
                        idx2 = int(k & 0xffffffff)
                        if idx1 and idx2:
                            yield idx1, idx2

    def overlaps():
        for D in range(depth - 1):
            print "depth", D
            for Seg in range(numsegs):
                labels1 = labels[Seg, D, :, :][...]
                labels1 <<= 32
                # all overlaps at next depth
                for Seg2 in range(numsegs):
                    labels2 = labels[Seg2, D + 1, :, :][...]
                    np.add(labels1, labels2, labels2)
                    keys, counts = pandas.lib.value_count_int64(labels2.ravel())
                    for k, c in zip(keys, counts):
                        idx1 = int(k >> 32)
                        idx2 = int(k & 0xffffffff)
                        if idx1 and idx2:
                            yield idx1, idx2, link_worth(float(areas[idx1]), float(areas[idx2]), float(c))

    print "Area counting took", int(time.time() - st), "seconds"

    return areas, exclusions(), overlaps()

def build_model(lpfile, areas, exclusions, overlaps):
    ##################################################
    # Generate the LP problem
    ##################################################
    print "Building MILP problem:"

    st = time.time()

    # Find all segments and sizes
    segments, segment_areas = zip(*sorted(areas.items()))
    segments = np.array(segments)
    segment_areas = np.array(segment_areas, dtype=float)

    # sanity checking - check all segments present
    assert np.all((segments[1:] - segments[:-1]) == 1)
    # needed for indexing
    assert segments[0] == 0

    # Build the LP
    num_segments = len(segments)
    print "  segments", num_segments
    objective = []
    segvars = []
    # Create variables for the segments and links
    obj_terms = ["%f seg_%d" % (weight, segidx)
                 for segidx, weight in enumerate(segment_worth(segment_areas))]

    print "finding links"
    # add links and link constraints
    uplinksets = {}
    downlinksets = {}
    link_to_segs = {}
    linknames = []
    for linkidx, (idx1, idx2, weight) in enumerate(overlaps):
        obj_terms += ["%f link_%d_%d" % (weight, idx1, idx2)]
        linknames += ["link_%d_%d" % (idx1, idx2)]
        uplinksets[idx1] = uplinksets.get(idx1, []) + ["link_%d_%d" % (idx1, idx2)]
        downlinksets[idx2] = downlinksets.get(idx2, []) + ["link_%d_%d" % (idx1, idx2)]

    num_links = linkidx

    print "Adding exclusions"
    excls = ["seg_%d + seg_%d <= 1" % (idx1, idx2) for idx1, idx2 in exclusions]


    print "found", num_links, "links"
    print "adding uplinks"
    uplink_excls = [("seg_%d - " % segidx) + " - ".join(links) + " >= 0"
                    for segidx, links in uplinksets.iteritems()]

    print "adding downlinks"
    downlink_excls = [("seg_%d - " % segidx) + " - ".join(links) + " >= 0"
                      for segidx, links in downlinksets.iteritems()]

    print "done"

    lpfile.write("Maximize\n")
    lpfile.write(" + ".join(obj_terms))
    lpfile.write("\n")
    lpfile.write("Subject To\n")
    lpfile.write("\n".join(excls))
    lpfile.write("\n")
    lpfile.write("\n".join(uplink_excls))
    lpfile.write("\n")
    lpfile.write("\n".join(downlink_excls))
    lpfile.write("\n")
    lpfile.write("Binary\n")
    lpfile.write("\n".join(("seg_%d" % idx) for idx in range(num_segments)))
    lpfile.write("\n")
    lpfile.write("\n".join(linknames))
    lpfile.write("\n")
    lpfile.write("End\n")

if __name__ == '__main__':
    segmentations = h5py.File(sys.argv[1])['cubesegs']

    ##################################################
    # compute all overlaps between multisegmentations
    ##################################################
    numsegs, depth, width, height = segmentations.shape

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
    chunking = list(segmentations.shape)
    chunking[0] = 1
    chunking[1] = 1
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
    build_model(open("theproblem.lp", "w"), areas, exclusions, overlaps)
    print "Building MILP too<k", int(time.time() - st), "seconds"

