import os
import sys
import time
import gc
import operator

import numpy as np
from scipy.ndimage.measurements import label
import h5py
import pandas

import cplex


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

label_offsets = {}
max_label = 0

@timed
def unique_labels(depth, seg, values):
    global max_label
    my_offset = label_offsets.setdefault((depth, seg), max_label)
    labels = label(values, output=np.int64)[0]
    labels += (labels > 0) * my_offset
    max_label = max(max_label, labels.max())
    return labels

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
            for Seg in range(numsegs):
                print D, Seg
                labels1 = labels[Seg, D, :, :][...]
                labels1 <<= 32

                l1flat = labels1.flat
                # all overlaps at this same depth
                for Seg2 in range(Seg + 1, numsegs):
                    labels2 = labels[Seg2, D, :, :][...]
                    np.add(labels1, labels2, labels2)
                    keys, counts = pandas.lib.value_count_int64(labels2.ravel())
                    for k in keys:
                        idx1 = int(k & 0xffff)
                        idx2 = int(k >> 32)
                        if idx1 and idx2:
                            yield idx1, idx2

    def overlaps():
        for D in range(depth - 1):
            for Seg in range(numsegs):
                print D, Seg
                labels1 = labels[Seg, D, :, :][...]
                labels1 <<= 32
                # all overlaps at next depth
                for Seg2 in range(numsegs):
                    labels2 = labels[Seg2, D + 1, :, :][...]
                    np.add(labels1, labels2, labels2)
                    keys, counts = pandas.lib.value_count_int64(labels2.ravel())
                    for k, c in zip(keys, counts):
                        idx1 = int(k & 0xffff)
                        idx2 = int(k >> 32)
                        if idx1 and idx2:
                            yield idx1, idx2, link_worth(float(areas[idx1]), float(areas[idx2]), float(c))

    print "Area counting took", int(time.time() - st), "seconds"

    return areas, exclusions(), overlaps()

def build_model(areas, exclusions, overlaps):
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
    model = cplex.Cplex()

    num_segments = len(segments)
    print "  segments", num_segments
    # Create variables for the segments and links
    model.variables.add(obj = segment_worth(segment_areas),
                        lb = [0] * num_segments,
                        ub = [1] * num_segments,
                        types = ["B"] * num_segments)

    print "Adding exclusions"
    # Add exclusion constraints
    for idx1, idx2 in exclusions:
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [idx1, idx2],
                                                                  val = [1, 1])],
                                     senses = "L",
                                     rhs = [1.1])

    print "finding links"
    # add links and link constraints
    uplinksets = {}
    downlinksets = {}
    link_to_segs = {}
    for idx1, idx2, weight in overlaps:
        linkidx = model.variables.get_num()
        model.variables.add(obj = [weight], lb = [0], ub = [1], types = "B", names = ['link%d' % linkidx])
        uplinksets[idx1] = uplinksets.get(idx1, []) + [linkidx]
        downlinksets[idx2] = downlinksets.get(idx2, []) + [linkidx]
        link_to_segs[linkidx] = (idx1, idx2)

    print "found", model.variables.get_num() - len(segments), "links"
    print "adding links"
    for segidx, links in uplinksets.iteritems():
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(segidx)] + links,
                                                                  val = [1] + [-1] * len(links))],
                                     senses = "G",
                                     rhs = [-0.05])

    for segidx, links in downlinksets.iteritems():
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(segidx)] + links,
                                                                  val = [1] + [-1] * len(links))],
                                     senses = "G",
                                     rhs = [-0.05])

    print "done"
    model.objective.set_sense(model.objective.sense.maximize)
    model.parameters.threads.set(1) 

    # model.write("theproblem.lp")
    return model, link_to_segs

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
    lf = h5py.File(output_path + '_partial', 'w')
    chunking = list(segmentations.shape)
    chunking[0] = 1
    chunking[1] = 1
    labels = lf.create_dataset('seglabels', segmentations.shape, dtype=np.int64, chunks=tuple(chunking), compression='gzip')
    for D in range(depth):
        for Seg in range(numsegs):
            labels[Seg, D, :, :] = unique_labels(D, Seg, segmentations[Seg, D, :, :][...])
    print "Labeling took", int(time.time() - st), "seconds"

    areas, exclusions, overlaps = count_overlaps(depth, numsegs, labels)
    num_segments = len(areas)

    st = time.time()
    model, links_to_segs = build_model(areas, exclusions, overlaps)
    print "Building MILP took", int(time.time() - st), "seconds"

    # free memory
    areas = exclusions = overlaps = None
    gc.collect()

    print "Solving"
    model.solve()
    print "Solving took", int(time.time() - st), "seconds"

    # Build the map from incoming label to linked labels
    segment_map = np.array(model.solution.get_values(0, num_segments - 1)).astype(np.int32)
    print segment_map.sum(), "active segments"
    segment_map *= np.arange(segment_map.shape[0])  # map every on segment to itself

    # Process links
    link_vars = np.array(model.solution.get_values()).astype(np.bool)
    link_vars[:num_segments] = 0
    print link_vars.sum(), "active links"
    for linkidx in np.nonzero(link_vars)[0]:
        l1, l2 = links_to_segs[linkidx]
        segment_map[l2] = l1  # link higher to lower

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

    # Apply map to labels
    for D in range(depth):
        for Seg in range(numsegs):
            labels[Seg, D, :, :] = segment_map[labels[Seg, D, :, :][...]]

    # Condense results
    out_labels = lf.create_dataset('labels', [depth, width, height], dtype=np.int32, chunks=tuple(chunking[1:]), compression='gzip')
    for D in range(depth):
        out_labels[D, :, :] = block_offset
        for Seg in range(numsegs):
            out_labels[D, :, :] += labels[Seg, D, :, :]

    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)

    os.rename(output_path+ '_partial', output_path)
    print "Successfully wrote", sys.argv[3]
