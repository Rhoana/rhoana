import os
import sys
import glob
import mahotas
import numpy as np
import fast64counter
from matplotlib.pyplot import figure
from matplotlib.pyplot import imshow


def load_stack(folder_name, ifrom=None, ito=None):

    stack = None
    input_files = sorted(glob.glob(os.path.join(folder_name, '*')))

    input_files = [i for i in input_files if not i.endswith('.db')]

    if ifrom is not None:
        if ito is not None:
            input_files = input_files[ifrom:ito]
        else:
            input_files = input_files[ifrom:ifrom+1]

    for i, file_name in enumerate(input_files):

        if file_name.endswith('h5') or file_name.endswith('hdf5'):
            infile = h5py.File(file_name)
            im = infile['/labels'][...]
        else:
            im = mahotas.imread(file_name)
            if len(im.shape) == 3 and im.shape[2] == 3:
                im = np.int32(im[ :, :, 0 ]) * 2**16 + np.int32(im[ :, :, 1 ]) * 2**8 + np.int32(im[ :, :, 2 ])
            elif len(im.shape) == 3 and im.shape[2] == 4:
                im = np.int32(im[ :, :, 0 ]) * 2**16 + np.int32(im[ :, :, 1 ]) * 2**8 + np.int32(im[ :, :, 2 ]) + np.int32(im[ :, :, 3 ]) * 2**24

        if stack is None:
            stack = np.zeros((len(input_files), im.shape[0], im.shape[1]), dtype=im.dtype)
            print 'Stack size={0}, dtype={1}.'.format(stack.shape, stack.dtype)
        stack[i,:,:] = im

    return stack


def filter_profiles_connected_components(img, min_seg_size):

    # Find 2D connected components (non-branching profiles)
    segments = mahotas.labeled.borders(img)==0
    segments[img==0] = 0
    profiles, nprofiles = mahotas.label(segments)

    profile_sizes = mahotas.labeled.labeled_size(profiles)
    too_small = np.nonzero(profile_sizes < min_seg_size)[0]

    too_small_area = np.sum(profile_sizes[too_small])
    ok_area = np.sum(profile_sizes) - too_small_area

    zero_segments = np.unique(profiles[img==0])
    ignore_segments = np.union1d(zero_segments, too_small)

    if len(ignore_segments > 0):
        #print "Ignoring {0} zero-segments and {1} small segments.".format(len(zero_segments), len(too_small))
        profiles, nprofiles = mahotas.labeled.relabel(mahotas.labeled.remove_regions(profiles, ignore_segments, inplace=True), inplace=True)
        profile_sizes = mahotas.labeled.labeled_size(profiles)

    return profiles, nprofiles, len(too_small), profile_sizes, too_small_area, ok_area


def split_profiles_connected_components(img, size_mask=None):

    # Find 2D connected components (non-branching profiles)
    segments = mahotas.labeled.borders(img)==0
    segments[img==0] = 0
    profiles, nprofiles = mahotas.label(segments)

    if size_mask is None:
        profile_sizes = mahotas.labeled.labeled_size(profiles)
    else:
        masked_profiles = profiles
        masked_profiles[size_mask] = 0
        profile_sizes = mahotas.labeled.labeled_size(masked_profiles)

    return profiles, nprofiles, profile_sizes

def compute_split_merge_errors_2d_image(gt_image, seg_image, min_seg_size=10, epsilon=0.4, display_results=True):

    error_image = np.zeros(gt_image.shape, dtype=np.int32)

    gt_profiles, gt_nprofiles, gt_profile_sizes = split_profiles_connected_components(gt_image)
    #seg_profiles, seg_nprofiles, seg_profile_sizes = split_profiles_connected_components(seg_image)
    # Ignore regions where the ground-truth label is zero
    seg_profiles, seg_nprofiles, seg_profile_sizes = split_profiles_connected_components(seg_image, gt_image==0)

    counter = fast64counter.ValueCountInt64()
    counter.add_values_pair32(gt_profiles.astype(np.int32).ravel(), seg_profiles.astype(np.int32).ravel())
    overlap_labels_gt, overlap_labels_seg, overlap_areas = counter.get_counts_pair32()

    gt_valid_regions = 0
    ngood_regions = 0
    nsplit_errors = 0
    nmerge_errors = 0
    nunique_merge_errors = 0
    nadjust_errors = 0
    goodRegionIndex = []

    good_area = 0
    gt_good_area = 0

    split_area = 0
    merge_area = 0
    adjust_area = 0
    gt_total_area = 0
    seg_total_area = 0

    repeat_merge_errors = {}

    for gt_id in range(1, gt_nprofiles+1):

        gt_size = float(gt_profile_sizes[gt_id])
        if gt_size < min_seg_size:
            continue

        gt_valid_regions += 1
        gt_total_area += gt_size

        valid_overlaps = np.nonzero(np.logical_and(overlap_labels_gt==gt_id,overlap_labels_seg!=0))[0]
        seg_counts = overlap_areas[valid_overlaps]
        #consider segs in order of size
        seg_order = np.argsort(seg_counts)[-1::-1]
        seg_counts = seg_counts[seg_order]
        seg_ids = overlap_labels_seg[valid_overlaps][seg_order]

        has_good_region = False
        has_merge_errors = False
        has_new_merge_errors = False
        has_split_errors = False
        has_adjust_errors = False

        for i, seg_id in enumerate(seg_ids):
            seg_size = float(seg_profile_sizes[seg_id])
            if seg_size < min_seg_size:
                continue

            overlap_size = seg_counts[i]
            seg_total_area += overlap_size

            gt_ratio = float(overlap_size) / gt_size
            seg_ratio = float(overlap_size) / seg_size
            match_gt = gt_ratio > 1 - epsilon   # big enough to match gt
            match_seg = seg_ratio > 1 - epsilon # big enough to match seg

            overlap_region = np.logical_and(gt_profiles == gt_id, seg_profiles == seg_id)

            if match_gt and match_seg:
                has_good_region = True
                goodRegionIndex.append(gt_id)
                error_image[overlap_region] = 1
                good_area += overlap_size
                break
            elif match_gt and not match_seg:
                # false positive (merge error)
                if not seg_id in repeat_merge_errors:
                    repeat_merge_errors[seg_id] = True
                    has_new_merge_errors = True
                has_merge_errors = True
                error_image[overlap_region] = 2
                merge_area += overlap_size
            elif not match_gt and match_seg:
                # false negative (split error)
                has_split_errors = True
                error_image[overlap_region] = 3
                split_area += overlap_size
            else:
                # both
                has_adjust_errors = True
                error_image[overlap_region] = 4
                adjust_area += overlap_size

        if has_good_region:
            ngood_regions += 1
            gt_good_area += gt_size
        elif has_merge_errors:
            nmerge_errors += 1
            if has_new_merge_errors:
                nunique_merge_errors += 1
        elif has_split_errors:
            nsplit_errors += 1
        elif has_adjust_errors:
            nadjust_errors += 1

    if display_results:

        # dx, dy = np.gradient(gt_image)
        # gt_boundaries = boundary = np.logical_or(dx!=0, dy!=0)
        # dx, dy = np.gradient(seg_image)
        # seg_boundaries = boundary = np.logical_or(dx!=0, dy!=0)

        # color_boundary_image = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
        # color_boundary_image[:,:,0] = seg_boundaries * 255
        # color_boundary_image[:,:,1] = gt_boundaries * 255
        # color_boundary_image[:,:,2] = seg_boundaries * 255

        # figure(figsize=(20,20))
        # imshow(color_boundary_image)

        color_error_image = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
        color_error_image[:,:,0] = np.logical_or(error_image == 2, error_image==4) * 255
        color_error_image[:,:,1] = (error_image == 1) * 255
        color_error_image[:,:,2] = np.logical_or(error_image == 3, error_image==4) * 255

        figure(figsize=(20,20))
        imshow(color_error_image)

        print '{0:8d} good (2d) regions    ={1:6.2f}% of regions ={2:6.2f}% of pixels.'.format(ngood_regions, float(ngood_regions) / gt_valid_regions * 100, float(gt_good_area) / gt_total_area * 100)
        print '{0:8d} split error regions  ={1:6.2f}% of regions ={2:6.2f}% of pixels.'.format(nsplit_errors, float(nsplit_errors) / gt_valid_regions * 100, float(split_area) / gt_total_area * 100)
        print '{0:8d} merge error regions  ={1:6.2f}% of regions ={2:6.2f}% of pixels.'.format(nmerge_errors, float(nmerge_errors) / gt_valid_regions * 100, float(merge_area) / gt_total_area * 100)
        print '{0:8d} Umerge error regions ={1:6.2f}% of regions ={2:6.2f}% of pixels.'.format(nunique_merge_errors, float(nunique_merge_errors) / gt_valid_regions * 100, float(merge_area) / gt_total_area * 100)
        print '{0:8d} adjust error regions ={1:6.2f}% of regions ={2:6.2f}% of pixels.'.format(nadjust_errors, float(nadjust_errors) / gt_valid_regions * 100, float(adjust_area) / gt_total_area * 100)

    return ngood_regions, nsplit_errors, nmerge_errors, nunique_merge_errors, nadjust_errors, gt_valid_regions, error_image


def compute_split_merge_errors_2d_stack(gt_folder, seg_folder, min_seg_size=10, epsilon=0.4, display_results=True):
    # Load the ground truth and segmented volumes

    gt_stack = load_stack(gt_folder)
    seg_stack = load_stack(seg_folder)

    ngood_regions_total = 0
    nsplit_errors_total = 0
    nmerge_errors_total = 0
    nunique_merge_errors_total = 0
    nadjust_errors_total = 0
    gt_valid_regions_total = 0

    for zi in range(seg_stack.shape[0]):

        print '  z={0}.'.format(zi)

        ngood_regions, nsplit_errors, nmerge_errors, nunique_merge_errors, nadjust_errors, gt_valid_regions, error_image = \
            compute_split_merge_errors_2d_image(gt_stack[zi,:,:], seg_stack[zi,:,:], min_seg_size, epsilon, display_results)

        ngood_regions_total += ngood_regions
        nsplit_errors_total += nsplit_errors
        nmerge_errors_total += nmerge_errors
        nunique_merge_errors_total += nunique_merge_errors
        nadjust_errors_total += nadjust_errors
        gt_valid_regions_total += gt_valid_regions

    print '\n----- Stack Summary -----'
    print '{0:4d} good (2d) regions    ={1:6.2f}% of total 2d regions'.format(ngood_regions_total, float(ngood_regions_total) / gt_valid_regions_total * 100)
    print '{0:4d} split error regions  ={1:6.2f}% of total 2d regions'.format(nsplit_errors_total, float(nsplit_errors_total) / gt_valid_regions_total * 100)
    print '{0:4d} merge error regions  ={1:6.2f}% of total 2d regions'.format(nmerge_errors_total, float(nmerge_errors_total) / gt_valid_regions_total * 100)
    print '{0:4d} Umerge error regions ={1:6.2f}% of total 2d regions'.format(nunique_merge_errors_total, float(nunique_merge_errors_total) / gt_valid_regions_total * 100)
    print '{0:4d} adjust error regions ={1:6.2f}% of total 2d regions'.format(nadjust_errors_total, float(nadjust_errors_total) / gt_valid_regions_total * 100)


if __name__ == "__main__":

    # Default parameters
    min_seg_size = 10
    epsilon = 0.4
    display_results = True

    if len(sys.argv) < 3:
        print "usage: python ComputeSplitMergeError.py gt_folder seg_folder [min_seg_size epsilon display]"
    else:
        if len(sys.argv) > 3:
            min_seg_size = int(sys.argv[3])
        if len(sys.argv) > 4:
            epsilon = float(sys.argv[4])
        if len(sys.argv) > 5:
            display_results = sys.argv[5].lower() in ("on", "yes", "y", "true", "t", "1")

        gt_folder = sys.argv[1]
        seg_folder = sys.argv[2]

        compute_split_merge_errors_2d_stack(gt_folder, seg_folder, min_seg_size, epsilon, display_results)
