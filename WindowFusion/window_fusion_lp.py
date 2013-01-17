import os
import sys
import time
import gc
import operator
from collections import defaultdict

import numpy as np
from scipy.ndimage.measurements import label as ndimage_label
import h5py
import pylab

import overlaps

DEBUG = True

##################################################
# Parameters
##################################################
size_compensation_factor = 0.9
chunksize = 256  # chunk size in the HDF5

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

def offset_labels(depth, seg, labels, offset):
    if offset == 0:
        return
    for xslice, yslice in overlaps.work_by_chunks(labels):
        l = labels[seg, depth, xslice, yslice][...]
        l[l > 0] += offset
        labels[seg, depth, xslice, yslice] = l

def build_model(lpfile, areas, exclusions, overlaps):
    ##################################################
    # Generate the LP problem
    ##################################################
    print "Building MILP problem:"

    st = time.time()

    # Build the LP
    num_segments = len(areas)
    print "  segments", num_segments
    objective = []
    segvars = []
    # Create variables for the segments and links
    obj_terms = ["%f seg_%d" % (weight, segidx)
                 for segidx, weight in enumerate(segment_worth(areas))]

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
    print "found", num_links, "links"

    print "Adding exclusions"
    overlap_excls = ["%s <= 1" % (" + ".join("seg_%d" for i in excl)) for excl in exclusions]

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
    lpfile.write("\n".join(overlap_excls))
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

    condense_labels = timed(overlaps.condense_labels)

    lf = h5py.File(output_path + '_partial', 'w')
    chunking = [1, 1, chunksize, chunksize]
    labels = lf.create_dataset('seglabels', segmentations.shape, dtype=np.int32, chunks=tuple(chunking), compression='gzip')
    total_regions = 0
    cross_D_offset = 0
    for D in range(depth):
        this_D_offset = 0
        for Seg in range(numsegs):
            temp, numregions = ndimage_label(segmentations[Seg, D, :, :][...], output=np.int32)
            labels[Seg, D, :, :] = temp
            offset_labels(D, Seg, labels, this_D_offset)
            this_D_offset += numregions
            total_regions += numregions
        condensed_count = condense_labels(D, numsegs, labels)
        print "Labeling depth %d: original %d, condensed %d" % (D, this_D_offset, condensed_count)
        for Seg in range (numsegs):
            offset_labels(D, Seg, labels, cross_D_offset)
        cross_D_offset += condensed_count
        # XXX - apply cross-D offset
    print "Labeling took", int(time.time() - st), "seconds, ", condense_labels.total_time, "in condensing"
    print cross_D_offset, "total labels", total_regions, "before condensing"

    if DEBUG:
        assert np.max(labels) == cross_D_offset 

    areas, exclusions, overlaps = overlaps.count_overlaps_exclusionsets(depth, numsegs, labels, link_worth)
    num_segments = len(areas)
    assert num_segments == cross_D_offset + 1  # areas includes an area for 0

    st = time.time()
    build_model(open("theproblem.lp", "w"), areas, exclusions, overlaps)
    print "Building MILP too<k", int(time.time() - st), "seconds"

