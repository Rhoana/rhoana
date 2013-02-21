from collections import defaultdict

import numpy as np
from scipy.ndimage.measurements import label as ndimage_label

import fast64counter

DEBUG = False

def work_by_chunks(dataset):
    xchunk, ychunk = dataset.chunks[-2:]
    for xbase in range(0, dataset.shape[-2], xchunk):
        for ybase in range(0, dataset.shape[-1], ychunk):
            yield slice(xbase, xbase + xchunk), slice(ybase, ybase + ychunk)

def condense_labels(depth, numsegs, labels):
    # project all labels at a given Z into a single plane, then merge any that
    # end up mapping to the same projected subregions (merge = remove one of
    # them)
    #
    # Regions in the presegmentations tend to be similar.  This step merges
    # segments that differ only near the boundaries.

    # work by chunks for speed.  Note that this approach splits projected
    # segments at chunk boundaries, but since we're just using them for
    # identifying similar regions, this is fine.
    overlap_counter = fast64counter.ValueCountInt64()
    sublabel_offset = 0
    for xslice, yslice in work_by_chunks(labels):
        chunklabels = [labels[S, depth, xslice, yslice][...] for S in range(numsegs)]
        projected = chunklabels[0] > 0
        for S in range(1, numsegs):
            projected &= chunklabels[S] > 0  # Mask out boundaries
        labeled_projected, num_found = ndimage_label(projected, output=np.int32)
        labeled_projected[labeled_projected > 0] += sublabel_offset
        sublabel_offset += num_found
        for sub in chunklabels:
            overlap_counter.add_values_pair32(labeled_projected.ravel(), sub.ravel())

    # Build original label to set of projected label map
    projected_labels, original_labels, areas = overlap_counter.get_counts_pair32()
    label_to_projected = defaultdict(set)
    for pl, ol in zip(projected_labels, original_labels):
        if ol and pl:
            label_to_projected[ol].add(pl)

    # Reverse the map
    projected_to_label = defaultdict(list)
    for ol, plset in label_to_projected.iteritems():
        projected_to_label[tuple(sorted(plset))].append(ol)

    # Build a remapper to remove merged labels
    remapper = np.arange(np.max(original_labels) + 1)
    for original_label_list in projected_to_label.itervalues():
        # keep the first, but zero the rest
        if len(original_label_list) > 1:
            remapper[original_label_list[1:]] = 0
#             for l in original_label_list:
#                 pylab.figure()
#                 for s in range(numsegs):
#                     mask = labels[s, depth, :, :] == l
#                     if np.any(mask):
#                         pylab.imshow(mask)
#                         pylab.title('label %d at depth %d' % (l, s))
#                         break
#             pylab.show()

    # pack the labels in the remapper
    final_label_count = np.sum(remapper > 0)
    remapper[0] = 1  # simplify next line
    remapper[remapper > 0] = np.arange(final_label_count + 1, dtype=np.int32)

    # remap the labels by chunk
    for xslice, yslice in work_by_chunks(labels):
        for S in range(numsegs):
            l = labels[S, depth, xslice, yslice]
            labels[S, depth, xslice, yslice] = remapper[l]

    if DEBUG:
        assert len(np.unique(labels[:, depth, ...].ravel())) == final_label_count + 1

    return final_label_count



def count_overlaps_exclusionsets(depth, numsegs, labels, link_worth):
    areacounter = fast64counter.ValueCountInt64()
    # Count areas of each label
    for xslice, yslice in work_by_chunks(labels):
        for D in range(depth):
            for Seg in range(numsegs):
                lbls = labels[Seg, D, xslice, yslice][...]
                areacounter.add_values_32(lbls.ravel())
    keys, areas = areacounter.get_counts()
    areas = areas[np.argsort(keys)]
    areas[0] = 0

    # sanity check
    assert np.all(np.sort(keys) == np.arange(len(keys)))

    def exclusions():
        for D in range(depth):
            print "exl depth", D
            excls = set()
            for xslice, yslice in work_by_chunks(labels):
                subimages = [labels[Seg, D, xslice, yslice][...].ravel() for Seg in range(numsegs)]
                excls.update(set(zip(*subimages)))
            # filter out zeros
            excls = set(tuple(i for i in s if i) for s in excls)
            for excl in excls:
                if len(excl) > 1:
                    yield excl

    def overlaps():
        overlap_areas = fast64counter.ValueCountInt64()
        for D in range(depth - 1):
            for xslice, yslice in work_by_chunks(labels):
                subimages_d1 = [labels[Seg, D, xslice, yslice][...].ravel() for Seg in range(numsegs)]
                subimages_d2 = [labels[Seg, D + 1, xslice, yslice][...].ravel() for Seg in range(numsegs)]
                for s1 in subimages_d1:
                    for s2 in subimages_d2:
                        overlap_areas.add_values_pair32(s1, s2)
        idxs1, idxs2, overlap_areas = overlap_areas.get_counts_pair32()
        mask = (idxs1 > 0) & (idxs2 > 0)
        idxs1 = idxs1[mask]
        idxs2 = idxs2[mask]
        overlap_areas = overlap_areas[mask]
        print len(idxs1), "Overlaps"
        for idx1, idx2, overlap_area in zip(idxs1, idxs2, overlap_areas):
            yield idx1, idx2, link_worth(float(areas[idx1]), float(areas[idx2]), float(overlap_area))

    return areas, exclusions(), overlaps()

