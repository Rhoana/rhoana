import sys
import h5py
import numpy as np
from scipy.ndimage.measurements import label
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import pulp
import time
from collections import Counter, defaultdict
import pandas


##################################################
# Parameters
##################################################
size_compensation_factor = 0.9


# weights for segments
def segment_worth(area):
    return area ** size_compensation_factor

def link_worth(area1, area2, area_overlap):
    min_area = min(area1, area2)
    max_fraction = area_overlap / max(area1, area2)
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

class LPProblem(object):
    def __init__(self):
        self.segments = {}  # segment label number to pulp variable
        self.links = {}  # pair of segment labels to pulp variable
        self.up_links = {}  # segment label to list of link variables
        self.down_links = {}  # segment label to list of link variables
        self.lp = pulp.LpProblem("window_fusion", pulp.LpMaximize)

    def add_overlaps(self, overlaps, same_depth=True):
        nonzero_rows = list(set(overlaps.nonzero()[0]))
        seg_areas = dict((r, overlaps[r, :].sum()) for r in nonzero_rows)

        # Create variables for each segment
        for segidx in seg_areas:
            self.add_segment(segidx, seg_areas[segidx])

        # Create links and/or constraints for each overlapping pair of segments
        for seg1_idx in seg_areas:
            if seg1_idx == 0:
                continue
            for seg2_idx in overlaps[seg1_idx, :].nonzero()[1]:
                if seg2_idx == 0:
                    continue
                if not same_depth:
                    self.add_link(seg1_idx, seg2_idx,
                                  seg_areas[seg1_idx], seg_areas[seg2_idx],
                                  overlaps[seg1_idx, seg2_idx])
                else:
                    self.add_exclusion(seg1_idx, seg2_idx)

    def add_segment(self, segidx, segarea):
        if segidx in self.segments:
            return
        segvar = pulp.LpVariable('seg_%d' % segidx, cat=pulp.LpBinary)
        self.segments[segidx] = segvar

        # add segment to objective function
        self.lp += segment_worth(segarea) * segvar

    def add_link(self,
                 seg1_idx, seg2_idx,
                 seg1_area, seg2_area,
                 overlap_area):
        if (seg1_idx, seg2_idx) in self.links or \
                (seg2_idx, seg1_idx) in self.links:
            return
        linkvar = pulp.LpVariable('link_%d_%d' % (seg1_idx, seg2_idx),
                                 cat=pulp.LpBinary)
        self.links[seg1_idx, seg2_idx] = linkvar

        # add link to objective function
        self.lp += link_worth(seg1_area, seg2_area, overlap_area) * linkvar

        if seg1_idx < seg2_idx:
            self.up_links[seg1_idx] = self.up_links.get(seg1_idx, []) + [linkvar]
            self.down_links[seg2_idx] = self.down_links.get(seg2_idx, []) + [linkvar]
        else:
            self.down_links[seg1_idx] = self.down_links.get(seg1_idx, []) + [linkvar]
            self.up_links[seg2_idx] = self.up_links.get(seg2_idx, []) + [linkvar]

        # XXX - if branching is turned on, we need a constraint that makes sure
        # both segments are activated if the link is activated

    def add_exclusion(self, seg1_idx, seg2_idx):
        # keep both segments from being activated at the same time
        self.lp += (self.segments[seg1_idx] + self.segments[seg2_idx]) <= 1

    def add_non_branching_constraints(self):
        # Every link is between two segment indexes.  We can tell which
        # direction (up or down) based on which segment index is lower.

        # every set of links can have only one active
        for segidx, links in self.up_links.iteritems():
            self.lp += pulp.LpAffineExpression(links) <= 1
        for segidx, links in self.down_links.iteritems():
            self.lp += pulp.LpAffineExpression(links) <= 1


if __name__ == '__main__':
    segmentations = h5py.File(sys.argv[1])['cubesegs']

    ##################################################
    # compute all overlaps between multisegmentations
    ##################################################
    numsegs, depth, width, height = segmentations.shape
    print segmentations.shape

    # ensure we can store all the labels we need to
    assert (width * height * depth * numsegs) < (2 ** 32 - 1), \
        "Cube too large.  Must be smaller than 2**32 - 1 voxels."

    largest_index = depth * numsegs * width * height

    lpprob = LPProblem()

    areas = defaultdict(int)

    st = time.time()

    # Compute all the labels for all the slices, making each unique as necessary
    print "Computing segment-to-segment overlaps"
    for D in range(depth)[:2]:
        for Seg in range(numsegs):
            print "Slice %d / %d, Segmentation %d / %d" % (D, depth, Seg, numsegs)
            print unique_labels.total_time, "of", time.time() - st
            print "Expected:", ((time.time() - st) * depth * numsegs) / (D * numsegs + Seg + 0.01)
            labels1 = unique_labels(D, Seg, segmentations[Seg, D, :, :][...])
            counts = pandas.lib.value_count_int64(labels1.ravel())
            labels1 <<= 32

            l1flat = labels1.flat
            # all overlaps at this same depth
            for Seg2 in range(Seg + 1, numsegs):
                labels2 = unique_labels(D, Seg2, segmentations[Seg2, D, :, :][...])
                np.add(labels1, labels2, labels2)
                counts = pandas.lib.value_count_int64(labels2.ravel())

            # all overlaps at next depth
            if D < depth - 1:
                for Seg2 in range(numsegs):
                    labels2 = unique_labels(D + 1, Seg2, segmentations[Seg2, D + 1, :, :][...])
                    np.add(labels1, labels2, labels2)
                    counts = pandas.lib.value_count_int64(labels2.ravel())


            print "  %d segments, %d links" % (len(lpprob.segments), len(lpprob.links))
