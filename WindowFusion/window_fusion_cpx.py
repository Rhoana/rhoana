import os
import sys
import time
import gc
import operator

import numpy as np
from scipy.ndimage.measurements import label as ndimage_label
import h5py

import cplex

import overlaps

DEBUG = False

##################################################
# Parameters
##################################################
size_compensation_factor = 0.9
chunksize = 128  # chunk size in the HDF5

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

def offset_labels(Z, seg, labels, offset):
    if offset == 0:
        return
    for xslice, yslice in overlaps.work_by_chunks(labels):
        l = labels[yslice, xslice, seg, Z][...]
        l[l > 0] += offset
        labels[yslice, xslice, seg, Z] = l

def build_model(areas, exclusions, links):
    ##################################################
    # Generate the LP problem
    ##################################################
    print "Building MILP problem:"

    st = time.time()

    # Build the LP
    model = cplex.Cplex()
    num_segments = len(areas)
    print "  segments", num_segments
    # Create variables for the segments and links
    model.variables.add(obj = segment_worth(areas),
                        lb = [0] * num_segments,
                        ub = [1] * num_segments,
                        types = ["B"] * num_segments)

    print "Adding exclusions"
    # Add exclusion constraints
    for ct, excl in enumerate(exclusions):
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(i) for i in excl],
                                                                  val = [1] * len(excl))],
                                     senses = "L",
                                     rhs = [1])
    print "  ", ct, "exclusions"

    print "finding links"
    # add links and link constraints
    uplinksets = {}
    downlinksets = {}
    link_to_segs = {}
    for idx1, idx2, weight in links:
        linkidx = model.variables.get_num()
        model.variables.add(obj = [weight], lb = [0], ub = [1], types = "B", names = ['link%d' % linkidx])
        uplinksets[idx1] = uplinksets.get(idx1, []) + [linkidx]
        downlinksets[idx2] = downlinksets.get(idx2, []) + [linkidx]
        link_to_segs[linkidx] = (idx1, idx2)

    print "found", model.variables.get_num() - num_segments, "links"
    print "adding links"
    for segidx, linklist in uplinksets.iteritems():
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(segidx)] + linklist,
                                                                  val = [1] + [-1] * len(linklist))],
                                     senses = "G",
                                     rhs = [0])

    for segidx, linklist in downlinksets.iteritems():
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(segidx)] + linklist,
                                                                  val = [1] + [-1] * len(linklist))],
                                     senses = "G",
                                     rhs = [0])

    print "done"
    model.objective.set_sense(model.objective.sense.maximize)
    model.parameters.threads.set(1) 
    model.parameters.mip.tolerances.mipgap.set(0.02)  # 2% tolerance
    # model.parameters.emphasis.memory.set(1)  # doesn't seem to help
    model.parameters.emphasis.mip.set(1)

    # model.write("theproblem.lp")
    return model, link_to_segs

if __name__ == '__main__':
    segmentations = h5py.File(sys.argv[1])['segmentations']

    ##################################################
    # compute all overlaps between multisegmentations
    ##################################################
    height, width, numsegs, numslices = segmentations.shape

    # ensure we can store all the labels we need to
    assert (height * width * numsegs * numslices) < (2 ** 31 - 1), \
        "Cube too large.  Must be smaller than 2**31 - 1 voxels."

    largest_index = numslices * numsegs * width * height

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
        print e
        pass

    condense_labels = timed(overlaps.condense_labels)

    lf = h5py.File(output_path + '_partial', 'w')
    chunking = [chunksize, chunksize, 1, 1]
    labels = lf.create_dataset('seglabels', segmentations.shape, dtype=np.int32, chunks=tuple(chunking), compression='gzip')
    total_regions = 0
    cross_Z_offset = 0
    for Z in range(numslices):
        this_slice_offset = 0
        for seg_idx in range(numsegs):
            temp, numregions = ndimage_label(segmentations[:, :, seg_idx, Z][...], output=np.int32)
            labels[:, :, seg_idx, Z] = temp
            offset_labels(Z, seg_idx, labels, this_slice_offset)
            this_slice_offset += numregions
            total_regions += numregions
        condensed_count = condense_labels(Z, numsegs, labels)
        print "Labeling depth %d: original %d, condensed %d" % (Z, this_slice_offset, condensed_count)
        for seg_idx in range (numsegs):
            offset_labels(Z, seg_idx, labels, cross_Z_offset)
        cross_Z_offset += condensed_count
        # XXX - apply cross-D offset
    print "Labeling took", int(time.time() - st), "seconds, ", condense_labels.total_time, "in condensing"
    print cross_Z_offset, "total labels", total_regions, "before condensing"

    if DEBUG:
        assert np.max(labels) == cross_Z_offset

    areas, exclusions, links = overlaps.count_overlaps_exclusionsets(numslices, numsegs, labels, link_worth)
    num_segments = len(areas)
    assert num_segments == cross_D_offset + 1  # areas includes an area for 0

    st = time.time()
    model, links_to_segs = build_model(areas, exclusions, links)
    print "Building MILP took", int(time.time() - st), "seconds"

    # free memory
    areas = exclusions = links = None
    gc.collect()

    print "Solving"
    model.solve()
    print "Solving took", int(time.time() - st), "seconds"

    # Build the map from incoming label to linked labels
    on_segments = np.array(model.solution.get_values(range(num_segments))).astype(np.bool)
    print on_segments.sum(), "active segments"
    segment_map = np.arange(num_segments, dtype=np.uint64)
    segment_map[~ on_segments] = 0

    if DEBUG:
        # Sanity check
        areas, exclusions, links = overlaps.count_overlaps_exclusionsets(numslices, numsegs, labels, link_worth)
        for excl in exclusions:
            assert sum(on_segments[s] for s in excl) <= 1

    # Process links
    link_vars = np.array(model.solution.get_values()).astype(np.bool)
    link_vars[:num_segments] = 0
    print link_vars.sum(), "active links"
    for linkidx in np.nonzero(link_vars)[0]:
        l1, l2 = links_to_segs[linkidx]
        assert on_segments[l1]
        assert on_segments[l2]
        segment_map[l2] = l1  # link higher to lower
        print "linked", l2, "to", l1

    # set background to 0
    segment_map[0] = 0
    # Compress labels
    next_label = 1
    for idx in range(1, len(segment_map)):
        if segment_map[idx] == idx:
            segment_map[idx] = next_label
            next_label += 1
        else:
            segment_map[idx] = segment_map[segment_map[idx]]

    assert (segment_map > 0).sum() == on_segments.sum()
    segment_map[segment_map > 0] |= block_offset

    for linkidx in np.nonzero(link_vars)[0]:
        l1, l2 = links_to_segs[linkidx]
        assert segment_map[l1] == segment_map[l2]

    # Condense results
    out_labels = lf.create_dataset('labels', [height, width, numslices], dtype=np.uint64,
                                   chunks=tuple(chunking[0], chunking[1], chunking[3]), compression='gzip')
    for Z in range(numslices):
        for seg_idx in range(numsegs):
            if (out_labels[:, :, Z][...].astype(bool) * segment_map[labels[:, :, seg_idx, Z]].astype(bool)).sum() != 0:
                badsegs = out_labels[:, :, Z][...].astype(bool) * segment_map[labels[:, :, seg_idx, Z]].astype(bool) != 0
                print "BAZ", out_labels[:, :, Z][badsegs], segment_map[labels[:, :, seg_idx, Z]][badsegs]
            out_labels[:, :, Z] |= segment_map[labels[:, :, seg_idx, Z]]

    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)

    os.rename(output_path+ '_partial', output_path)
    print "Successfully wrote", sys.argv[3]
