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
from collections import defaultdict


##################################################
# Parameters
##################################################
size_compensation_factor = 0.9
chunksize = 256

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
            print "exl depth", D
            excls = set()
            for xpos in range(0, labels.shape[2], chunksize):
                for ypos in range(0, labels.shape[3], chunksize):
                    subimages = [labels[Seg, D, xpos:(xpos + chunksize), ypos:(ypos + chunksize)][...].ravel() for Seg in range(numsegs)]
                    excls.update(set(zip(*subimages)))
            for excl in excls:
                yield excl

    def overlaps():
        for D in range(depth - 1):
            print "depth", D
            overlap_areas = defaultdict(int)
            for xpos in range(0, labels.shape[2], chunksize):
                for ypos in range(0, labels.shape[3], chunksize):
                    subimages_d1 = [labels[Seg, D, xpos:(xpos + chunksize), ypos:(ypos + chunksize)][...].ravel() for Seg in range(numsegs)]
                    subimages_d2 = [labels[Seg, D + 1, xpos:(xpos + chunksize), ypos:(ypos + chunksize)][...].ravel() for Seg in range(numsegs)]
                    for s1 in subimages_d1:
                        for s2 in subimages_d2:
                            keys, counts = pandas.lib.value_count_int64(((s1 << 32) + s2).ravel())
                            idxs1 = (keys >> 32)
                            idxs2 = (keys & 0xffffffff)
                            mask = (idxs1 > 0) & (idxs2 > 0)
                            for idx1, idx2, c in zip(idxs1[mask], idxs2[mask], counts[mask]):
                                overlap_areas[idx1, idx2] += c
            for (idx1, idx2), c in overlap_areas.iteritems():
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
    model.variables.add(lb = [0] * num_segments,
                        ub = [1] * num_segments,
                        types = ["B"] * num_segments)

    print "Adding exclusions"
    # Add exclusion constraints
    for exclusion_set in exclusions:
        indices = [int(i) for i in exclusion_set if i > 0]
        if indices:
            model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = indices,
                                                                      val = [1] * len(indices))],
                                         senses = "L",
                                         rhs = [1])

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
                                     rhs = [0])

    for segidx, links in downlinksets.iteritems():
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(segidx)] + links,
                                                                  val = [1] + [-1] * len(links))],
                                     senses = "G",
                                     rhs = [0])

    print "done"
    model.objective.set_sense(model.objective.sense.maximize)
    model.parameters.threads.set(1) 
    model.parameters.mip.tolerances.mipgap.set(0.02)  # 2% tolerance
    model.parameters.mip.display = 4  # noisy about root lp relaxation

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

    try:
        lf = h5py.File(output_path, 'r')
        if 'labels' in lf.keys():
            print "Output already generated"
            lf.close()
            sys.exit(0)
    except Exception, e:
        pass

    lf = h5py.File(output_path + '_partial', 'w')
    chunking = (1, 1, chunksize, chunksize)
    # We have to ues int64 for the pandas hash table implementation
    labels = lf.create_dataset('seglabels', segmentations.shape, dtype=np.int64, chunks=chunking, compression='gzip')
    offset = 0
    for D in range(depth):
        for Seg in range(numsegs):
            temp = unique_labels(D, Seg, segmentations[Seg, D, :, :][...], offset)
            labels[Seg, D, :, :] = temp
            offset = temp.max()
            assert offset < 2**31
    print "Labeling took", int(time.time() - st), "seconds"

    areas, exclusions, overlaps = count_overlaps(depth, numsegs, labels)
    num_segments = len(areas)
    print num_segments, offset
    assert num_segments == offset + 1  # areas includes an area for 0

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
    on_segments = np.array(model.solution.get_values(range(num_segments))).astype(np.bool)
    print on_segments.sum(), "active segments"
    segment_map = np.arange(num_segments, dtype=np.uint64)
    segment_map[~ on_segments] = 0

    # Sanity check
    areas, exclusions, overlaps = count_overlaps(depth, numsegs, labels)
    for s1, s2 in exclusions:
        assert not (on_segments[s1] and on_segments[s2])

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
    out_labels = lf.create_dataset('labels', [depth, width, height], dtype=np.uint64, chunks=chunking[1:], compression='gzip')
    for D in range(depth):
        for Seg in range(numsegs):
            assert (out_labels[D, :, :][...].astype(bool) * segment_map[labels[Seg, D, :, :]].astype(bool)).sum() == 0
            out_labels[D, :, :] |= segment_map[labels[Seg, D, :, :]]

    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)

    os.rename(output_path+ '_partial', output_path)
    print "Successfully wrote", sys.argv[3]
