import os
import sys
import time
import gc
import operator

import numpy as np
from scipy.ndimage.measurements import label
import h5py
import pandas
import pycpx


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
    exclusions = {}
    overlaps = {}

    st = time.time()

    # Count areas of each label
    for D in range(depth):
        for Seg in range(numsegs):
            lbls = labels[Seg, D, :, :][...]
            keys, counts = pandas.lib.value_count_int64(lbls.ravel())
            areas.update(zip(keys, counts))

    # Compute all the labels for all the slices, making each unique as necessary
    print "Computing segment-to-segment overlaps (%d slices, %d multisegmentations)" % (depth, numsegs)
    for D in range(depth):
        for Seg in range(numsegs):
            labels1 = labels[Seg, D, :, :][...]
            labels1 <<= 32

            l1flat = labels1.flat
            # all overlaps at this same depth
            for Seg2 in range(Seg + 1, numsegs):
                labels2 = labels[Seg2, D, :, :][...]
                np.add(labels1, labels2, labels2)
                keys, counts = pandas.lib.value_count_int64(labels2.ravel())
                exclusions.update(zip(keys, counts))

            # all overlaps at next depth
            if D < depth - 1:
                for Seg2 in range(numsegs):
                    labels2 = labels[Seg2, D + 1, :, :][...]
                    np.add(labels1, labels2, labels2)
                    keys, counts = pandas.lib.value_count_int64(labels2.ravel())
                    overlaps.update(zip(keys, counts))
    print "Overlaps took", int(time.time() - st), "seconds"

    return areas, exclusions, overlaps

def build_model(areas, exclusions, overlaps, bottom_segments):
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

    # Find all overlaps and overlap sizes
    links = np.array(overlaps.keys())
    link_areas = np.array(overlaps.values(), dtype=float)

    # find the segment indices for the links
    link_1st_segidx = (links >> 32)
    link_2nd_segidx = (links & 0xffffffff)
    # Filter links to/from background pixels
    mask = (link_1st_segidx > 0) & (link_2nd_segidx > 0)
    link_1st_segidx = link_1st_segidx[mask]
    link_2nd_segidx = link_2nd_segidx[mask]
    link_areas = link_areas[mask]
    # fetch areas
    link_1st_area = segment_areas[link_1st_segidx]
    link_2nd_area = segment_areas[link_2nd_segidx]

    # Build the LP
    model = pycpx.CPlexModel(verbosity=3)

    print "  segments", len(segments)
    print "  links", len(link_1st_segidx)
    # Create variables for the segments and links
    segment_vars = model.new(len(segments), vtype='binary')
    link_vars = model.new(len(link_1st_segidx), vtype='binary')

    # set up the objective
    objective = segment_vars.mult(segment_worth(segment_areas)).sum() + \
        link_vars.mult(link_worth(link_1st_area, link_2nd_area, link_areas)).sum()

    print "  overlap constraints", len(exclusions)
    # add overlap exclusion constraints
    for pair in exclusions:
        # can't be np.int64 if we're indexing CPlexModel variables
        idx1 = int(pair >> 32)
        idx2 = int(pair & 0xffffffff)
        if idx1 and idx2:
            model.constrain(segment_vars[idx1, 0] + segment_vars[idx2, 0] <= 1)

    print "  link constraints", len(link_1st_segidx)
    # add link constraints:
    # - no more than one link to any single segment active.
    # - if a link is active, the segment must be active.
    uplinks = {}
    downlinks = {}
    for linkidx, segidx in enumerate(link_1st_segidx):
        uplinks[segidx] = uplinks.get(segidx, []) + [linkidx]
    for linkidx, segidx in enumerate(link_2nd_segidx):
        downlinks[segidx] = downlinks.get(segidx, []) + [linkidx]
    for segidx, links in uplinks.iteritems():
        model.constrain(segment_vars[int(segidx), 0] >= sum(link_vars[l] for l in links))
    for segidx, links in downlinks.iteritems():
        model.constrain(segment_vars[int(segidx), 0] >= sum(link_vars[l] for l in links))

    # Initial solution is all finest segmentations activated
    initial_segment_values = np.zeros(segment_vars.shape, dtype=int)
    initial_segment_values[list(bottom_segments)] = 1
    initial_link_values = np.zeros(link_vars.shape, dtype=int)
    starting_dict = {segment_vars : initial_segment_values, link_vars : initial_link_values}
    return model, objective, starting_dict, segment_vars, link_vars, np.vstack((link_1st_segidx, link_2nd_segidx)).T

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
    output_path = sys.argv[2]
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

    st = time.time()
    bottom_segments = reduce(operator.ior, [set(labels[0, D, :, :].flat) for D in range(depth)])
    model, objective, start, segment_vars, link_vars, link_segments_pairs = build_model(areas, exclusions, overlaps, bottom_segments)
    print "Building MILP took", int(time.time() - st), "seconds"

    # free memory
    areas = exclusions = overlaps = None
    gc.collect()

    print "Solving"
    st = time.time()
    model.maximize(objective, starting_dict=start)
    print "Solving took", int(time.time() - st), "seconds"

    # Build the map from incoming label to linked labels
    segment_map = model[segment_vars].flatten().astype(np.int32)  # all on segments
    segment_map *= np.arange(segment_map.shape[0])  # map every on segment to itself
    active_links = model[link_vars].flatten()
    for l1, l2 in link_segments_pairs[active_links > 0, :]:
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
        out_labels[D, :, :] = 0
        for Seg in range(numsegs):
            out_labels[D, :, :] += labels[Seg, D, :, :]

    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)

    os.rename(output_path+ '_partial', output_path)
    print "Successfully wrote", sys.argv[2]
