import sys
import numpy as np
import scipy
import scipy.io
import scipy.ndimage
import mahotas
import math
import h5py
import time
import pymaxflow
import timer
import os

input_path = sys.argv[1]
output_path = sys.argv[2]

input_hdf5 = h5py.File(input_path, 'r')

n_orientations = 8
orientation_step = 180.0 / n_orientations

## Open the input images
norm_image = input_hdf5['original']
norm_image = norm_image - np.min(norm_image)
norm_image = norm_image / np.max(norm_image)

imshape = norm_image.shape

orient_image = np.zeros([imshape[0], imshape[1], n_orientations], dtype=np.float32)

prob_dset = input_hdf5['probabilities']
prob_image = prob_dset[...]

for oi in range(n_orientations):
    orientation_angle = int(orientation_step * oi)
    orient_image[:,:,oi] = input_hdf5['membrane_19_3_{0}'.format(orientation_angle)]

input_hdf5.close()

## Convert to reward orientations
orient_image = 1 - orient_image

## Take the maximum angle within each 45 degree window
orient_image_round = np.zeros([imshape[0], imshape[1], 4])
for round_oi in range(4):
    round_angle = 45.0 * round_oi
    for oi in range(n_orientations):
        orientation_angle = orientation_step * oi
        angle_difference = min(abs(round_angle - orientation_angle), abs(round_angle - orientation_angle + 180), abs(round_angle - orientation_angle - 180))
        print "Angles {0}, {1}, difference = {2}".format(round_angle, orientation_angle, angle_difference)
        if angle_difference <= 45.0 / 2:
            orient_image_round[:,:,round_oi] = np.maximum(orient_image_round[:,:,round_oi], orient_image[:,:,oi])


### Threshold and Gap Completion settings - loop over these values

##Threshold
#probability_thresholds = list(np.arange(0.21, 0.51, 0.01))
#smoothing_factors = [0.6]
#gap_completion_factors = [0.05]

##Gap completion
probability_thresholds = [0.5]
smoothing_factors = [0.2]
gap_completion_factors = list(np.arange(0.008, 0.24, 0.008))
gap_completion_factors.reverse()


##Single cut example
#probability_thresholds = [0.3]
#smoothing_factors = [0.6]
#gap_completion_factors = [0.05]

## Connectivity matrix settings
orientation_kernel_size = 49
orientation_line_thickness = 3
probability_offset = 1

## Smoothing settings
image_sigma = 1
prob_sigma = 1.1

## Tidy up settings
blur_radius = 2;
y,x = np.ogrid[-blur_radius:blur_radius+1, -blur_radius:blur_radius+1]
disc = x*x + y*y <= blur_radius*blur_radius
blur_sigma = 3

extra_cellular_dist = 10
y,x = np.ogrid[-extra_cellular_dist:extra_cellular_dist+1, -extra_cellular_dist:extra_cellular_dist+1]
min_disc = x*x + y*y <= extra_cellular_dist*extra_cellular_dist

blur_prob_scale = 2**16-1
blur_prob = (scipy.ndimage.gaussian_filter(prob_image, blur_sigma) * blur_prob_scale).astype(np.uint16)

## Calculate the graph cut connectivity matrices

## adjacency_matrix holds the i, j distance values for each direction
## first calculate as 8 images, one for each direction

## directions start at 3 o'clock and rotate clockwise
directions = [[1, 0], [1, 1], [0, 1], [-1, 1],
              [-1, 0], [-1, -1], [0, -1], [1, -1]]
smooth_distances = np.zeros([imshape[0], imshape[1], len(directions)], dtype=np.float32)
gap_completion_distances = np.zeros([imshape[0], imshape[1], len(directions)], dtype=np.float32)

# used for neighbor computations
coords = np.ogrid[0:imshape[0],
                  0:imshape[1]]
node_indices = np.arange(norm_image.size, dtype=np.int32).reshape(imshape)

def shift_coords(c, dir):
    return (c[0] + dir[0], c[1] + dir[1])

def validate_and_broadcast(*args):
    valid_i_mask = args[0][0] > 0
    valid_j_mask = args[0][1] > 0
    for icoords, jcoords in args:
        np.logical_and(valid_i_mask, icoords >= 0, out=valid_i_mask)
        np.logical_and(valid_j_mask, jcoords >= 0, out=valid_j_mask)
        np.logical_and(valid_i_mask, icoords < imshape[0], out=valid_i_mask)
        np.logical_and(valid_j_mask, jcoords < imshape[1], out=valid_j_mask)

    return [np.broadcast_arrays(icoords[valid_i_mask].reshape((-1, 1)),
                                jcoords[valid_j_mask].reshape((1, -1))) \
                for (icoords, jcoords) in args]

n_segmentations = len(probability_thresholds) * len(smoothing_factors) * len(gap_completion_factors)
segmentation_count = 0

# create the output in a temporary file
temp_path = output_path + '_tmp'
out_hdf5 = h5py.File(temp_path, 'w')
segmentations = out_hdf5.create_dataset('segmentations',
                                        (imshape[0], imshape[1], n_segmentations),
                                        dtype=np.bool,
                                        chunks=(256, 256, 1),
                                        compression='gzip')

# copy the probabilities for future use
probs_out = out_hdf5.create_dataset('probabilities',
                                    prob_image.shape,
                                    dtype = prob_image.dtype,
                                    chunks = prob_dset.chunks,
                                    compression='gzip')
probs_out[...] = prob_image

for di, direction in enumerate(directions):

    print 'Direction {0},{1}'.format(direction[0], direction[1]);

    shifted = norm_image

    if direction[0] != 0:
        shifted = np.roll(shifted, direction[0], 0);
        if direction[0] == 1:
            shifted[0,:] = 0
        else:
            shifted[-1,:] = 0

    if direction[1] != 0:
        shifted = np.roll(shifted, direction[1], 1);
        if direction[1] == 1:
            shifted[:,0] = 0
        else:
            shifted[:,-1] = 0

    ## Smoothing fuction on raw image
    smooth_distances[:,:,di] = 1 / (image_sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * np.power(norm_image - shifted, 2) / (image_sigma**2));

    ## Directional smoothing function on probabilities
    step_distance = np.sqrt(np.power(direction[0], 2) + np.power(direction[1], 2))
    gap_completion_distances[:,:,di] = abs(orient_image_round[:,:,di % 4]) * np.exp(-0.5 * np.power(prob_image - probability_offset, 2) / (prob_sigma**2)) / step_distance


## Loop over threshold / smoothing / gap completion settings

main_st = time.time()

for probability_threshold in probability_thresholds:

    ## terminal_matrix holds +ve weights in column 1 and -ve weights in column 2

    prob_image_vector = prob_image.ravel()

    positive_vals = prob_image_vector >= probability_threshold

    terminal_matrix_prob = np.zeros([prob_image_vector.size, 2], dtype=np.float32)

    indices = np.nonzero(positive_vals)[0]
    terminal_matrix_prob[indices, 0] = 2 * (prob_image_vector[indices] - probability_threshold)

    indices = np.nonzero(np.logical_not(positive_vals))
    terminal_matrix_prob[indices, 1] = -2 * (prob_image_vector[indices] - probability_threshold)

    terminal_matrix_smooth = np.zeros([prob_image_vector.size, 2], dtype=np.float32)
    terminal_matrix_smooth[:,0] = np.sum(smooth_distances, axis=2).ravel()
    terminal_matrix_smooth[:,1] = np.sum(smooth_distances, axis=2).ravel()

    terminal_matrix_gap_completion = np.zeros([prob_image_vector.size, 2], dtype=np.float32)
    terminal_matrix_gap_completion[:,0] = np.sum(gap_completion_distances, axis=2).ravel()

    #
    # Graph simplification.
    #
    # Consider a node n, and its 8 neighbors M.
    # If all of n and M have source_sink_cap > 0, and
    # sum(source_sink_cap[M]) > sum(outflow(M+{n})),
    #      where outflow(M+{n}) is flow from M except to n or other members of M,
    # then N can be ignored in the maxflow computation.
    # Similarly for inflow
    #
    # Compute neighborhood in/outflow matrices, used in graph simplification.
    with timer.Timer("inflow/outflow precomputation"):
        outflow_smooth = np.zeros(imshape)
        outflow_gap = np.zeros(imshape)
        inflow_smooth = np.zeros(imshape)
        inflow_gap = np.zeros(imshape)
        for di1, dir1 in enumerate(directions):

            for di2, dir2 in enumerate(directions):
                dotprod = ((dir1[0] * dir2[0]) +
                           (dir1[1] * dir2[1]))
                if (dir1[0] and dir1[1]):
                    # corner: good if no backtracking
                    valid_pair = (dotprod >= 0)
                else:
                    # side: good if forward progress
                    valid_pair = (dotprod == 1)
                if not valid_pair:
                    continue

                intermediate_coords = shift_coords(coords, dir1)
                exterior_coords = shift_coords(intermediate_coords, dir2)
                center_coords, intermediate_coords, exterior_coords = \
                    validate_and_broadcast(coords, intermediate_coords, exterior_coords)

                outflow_smooth[center_coords] += smooth_distances[:, :, di2][intermediate_coords]
                outflow_gap[center_coords] += gap_completion_distances[:, :, di2][intermediate_coords]
                di2reverse = (di2 + 4) % 8
                inflow_smooth[center_coords] += smooth_distances[:, :, di2reverse][exterior_coords]
                inflow_gap[center_coords] += gap_completion_distances[:, :, di2reverse][exterior_coords]

    for smoothing_factor in smoothing_factors:
        for gap_completion_factor in gap_completion_factors:
            with timer.Timer("setup"):

                flow_graph = pymaxflow.PyGraph(terminal_matrix_prob.shape[0],
                                               8 * terminal_matrix_prob.shape[0])

                flow_graph.add_node(terminal_matrix_prob.shape[0]);

                ## Load the terminal matrix
                terminal_matrix = terminal_matrix_prob + \
                    terminal_matrix_smooth * smoothing_factor + \
                    terminal_matrix_gap_completion * gap_completion_factor

                flow_graph.add_tweights_vectorized(node_indices.ravel(),
                                                   terminal_matrix[:,0].ravel(),
                                                   terminal_matrix[:,1].ravel())

                # compute the source/sink difference for the nodes
                source_sink_cap = terminal_matrix[:,0].flatten() - \
                    terminal_matrix[:,1].flatten()
                source_sink_cap = source_sink_cap.reshape(imshape)

                combined_distances = smooth_distances * smoothing_factor + \
                    gap_completion_distances * gap_completion_factor
                combined_outflow = outflow_smooth * smoothing_factor + \
                    outflow_gap * gap_completion_factor
                combined_inflow = inflow_smooth * smoothing_factor + \
                    inflow_gap * gap_completion_factor


                # Find ignorable nodes - those where full 3x3 neighborhood has
                # source_sink_cap > 0 and sum of source_sink_cap in
                # neighborhood (except center) is greater than neighborhood
                # outflow.  Similar for inflow.
                hollow = np.ones((3, 3))
                hollow[1, 1] = 0
                ignorable = np.logical_and(mahotas.erode(source_sink_cap > 0, np.ones((3,3))),
                                           (mahotas.convolve(source_sink_cap, hollow, mode='ignore') >
                                            combined_outflow))
                np.logical_or(ignorable,
                              np.logical_and(mahotas.erode(source_sink_cap < 0, np.ones((3,3))),
                                             (mahotas.convolve(source_sink_cap, hollow, mode='ignore') <
                                              - combined_inflow)),  # careful with signs
                              out=ignorable)

                print "Can ignore {0} of {1}".format(np.sum(ignorable), ignorable.size)

                ## Load the adjacency matrix
                for di, direction in enumerate(directions[:4]):
                    dest_coords = shift_coords(coords, direction)

                    source_coords, dest_coords = validate_and_broadcast(coords, dest_coords)
                    mask = np.logical_or(ignorable[source_coords],
                                         ignorable[dest_coords])
                    source_nodes = node_indices[source_coords][~ mask].ravel()
                    dest_nodes = node_indices[dest_coords][~ mask].ravel()

                    forward_distance = np.float32(combined_distances[:, :, di].flatten()[source_nodes])
                    backward_distance = np.float32(combined_distances[:, :, di + 4].flatten()[dest_nodes])

                    flow_graph.add_edge_vectorized(source_nodes, dest_nodes, forward_distance, backward_distance)

            ## Compute the max flow

            print "Running maxflow for probability threshold {0}, smoothing factor {1}, gap completion factor {2}.".format(probability_threshold, smoothing_factor, gap_completion_factor)
            with timer.Timer("maxflow"):
                print "Flow is {0}".format(flow_graph.maxflow())

            labels = flow_graph.what_segment_vectorized()
            labels = labels.reshape(imshape)

            with timer.Timer("close/open"):
                labels = mahotas.morph.close(labels.astype(np.bool), disc)
                labels = mahotas.morph.open(labels.astype(np.bool), disc)

            ## Use blurred probabilities and watershed instead of region growing
            with timer.Timer("label2"):
                seeds,_ = mahotas.label(labels==1)

            with timer.Timer("watershed"):
                ws = mahotas.cwatershed(blur_prob, seeds)

            with timer.Timer("gradient2"):
                dx, dy = np.gradient(ws)
                ws_boundary = np.logical_or(dx!=0, dy!=0)

            ## Identify possible extra-cellular space - distance method
            #extra_cellular = np.logical_and(mahotas.distance(labels==0) > 100, seeds == np.min(seeds))
            #extra_cellular = mahotas.morph.close(extra_cellular.astype(np.bool), disc)
            #extra_cellular = mahotas.morph.open(extra_cellular.astype(np.bool), disc)
            #extra_cellular_indices = np.nonzero(extra_cellular)

            ## Identify possible extra-cellular space - minima method
            with timer.Timer("extra_cellular"):
                with timer.Timer("    (sub)min"):
                    rmin = mahotas.regmin(blur_prob, min_disc)

                extra_cellular = np.logical_and(rmin, seeds == np.min(seeds))
                extra_cellular_indices = np.nonzero(extra_cellular)

            extra_cellular_id = np.max(seeds)+1
            seeds[extra_cellular_indices] = extra_cellular_id

            with timer.Timer("second watershed"):
                ws = mahotas.cwatershed(blur_prob, seeds)
            dx, dy = np.gradient(ws)
            ws_boundary = np.logical_or(dx!=0, dy!=0)

            ## Optional - mark the extra-cellular space as boundary
            #ws_boundary[np.nonzero(ws == extra_cellular_id)] = 1

            segmentations[:,:,segmentation_count] = ws_boundary > 0
            segmentation_count = segmentation_count + 1
            print "Segmentation {0} produced aftert {1} seconds".format(segmentation_count, int(time.time() - main_st))



# move to final destination
out_hdf5.close()
# move to final location
if os.path.exists(output_path):
    os.unlink(output_path)
os.rename(temp_path, output_path)
print "Success"
