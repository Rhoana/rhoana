import sys
import h5py
import numpy as np
from scipy.ndimage.measurements import label
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import pulp
import time
from collections import Counter, defaultdict
import pandas
import pycpx
import gc
import operator


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
    for D in range(depth):
        for Seg in range(numsegs):
            lbls = labels[Seg, D, :, :][...]
            keys, counts = pandas.lib.value_count_int64(lbls.ravel())
            areas.update(zip(keys, counts))
        print D

    st = time.time()

    # Compute all the labels for all the slices, making each unique as necessary
    print "Computing segment-to-segment overlaps"
    for D in range(depth):
        print "Slice %d / %d" % (D, depth), time.time() - st
        print "Expected:", ((time.time() - st) * depth * numsegs) / (D * numsegs + 0.01)
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

    return areas, exclusions, overlaps

def build_model(areas, exclusions, overlaps, bottom_segments):
    ##################################################
    # Generate the LP problem
    ##################################################
    print "Building problem:", len(areas), "segments,", len(exclusions), \
        "exclusions,", len(overlaps), "overlaps, (all approx.)"

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

    # Create variables for the segments and links
    segment_vars = model.new(len(segments), vtype='binary')
    link_vars = model.new(len(link_1st_segidx), vtype='binary')

    # set up the objective
    objective = segment_vars.mult(segment_worth(segment_areas)).sum() + \
        link_vars.mult(link_worth(link_1st_area, link_2nd_area, link_areas)).sum()

    print "overlap constraints", len(exclusions)
    # add overlap exclusion constraints
    for pair in exclusions:
        # can't be np.int64 if we're indexing CPlexModel variables
        idx1 = int(pair >> 32)
        idx2 = int(pair & 0xffffffff)
        if idx1 and idx2:
            model.constrain(segment_vars[idx1, 0] + segment_vars[idx2, 0] <= 1)

    print "link constraints", len(link_1st_segidx)
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
    import pdb
    pdb.set_trace()
    starting_dict = dict((s, 0) for s in segment_vars)
    starting_dict.update((l, 0) for l in link_vars)
    starting_dict.update((segment_vars[int(sidx), 0], 1) for sidx in bottom_segments if sidx > 0)
    return model, objective, starting_dict

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
    lf = h5py.File('temp.hdf5', 'w')
    chunking = list(segmentations.shape)
    chunking[0] = 1
    chunking[1] = 1
    labels = lf.create_dataset('labels', segmentations.shape, dtype=np.int64, chunks=tuple(chunking))
    for D in range(depth):
        for Seg in range(numsegs):
            labels[Seg, D, :, :] = unique_labels(D, Seg, segmentations[Seg, D, :, :][...])

    areas, exclusions, overlaps = count_overlaps(depth, numsegs, labels)

    bottom_segments = reduce(operator.ior, [set(labels[0, D, :, :].flat) for D in range(depth)])
    model, objective, start = build_model(areas, exclusions, overlaps, bottom_segments)
    # free memory
    areas = exclusions = overlaps = None
    gc.collect()

    print "solving"
    model.maximize(objective, starting_dict=start)
