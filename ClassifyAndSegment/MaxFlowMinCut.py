import numpy as np
import scipy
import scipy.io
import scipy.ndimage
import mahotas
import math
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

import pymaxflow

#plt.ion()

input_image_file = 'C:\\dev\\datasets\\conn\\main_dataset\\ac3train\\input_images\\I00005_image.tif'
input_prob_file = 'C:\\dev\\datasets\\conn\\main_dataset\\ac3train\\forest_prob_adj\\I00005_image.mat'
input_orientation_file = 'C:\\dev\\datasets\\conn\\main_dataset\\ac3train\\example_prediction.hdf5'

n_orientations = 8
orientation_step = 180.0 / n_orientations

## Open the input images
norm_image = np.array(mahotas.imread(input_image_file), dtype=np.float32)
norm_image = norm_image - np.min(norm_image)
norm_image = norm_image / np.max(norm_image)

orient_image = np.zeros([norm_image.shape[0], norm_image.shape[1], n_orientations], dtype=np.float32)

features_h5 = h5py.File(input_orientation_file, 'r')

prob_image = np.float32(features_h5['probabilities'])

for oi in range(n_orientations):
    orientation_angle = int(orientation_step * oi)
    orient_image[:,:,oi] = features_h5['membrane_19_3_{0}'.format(orientation_angle)]

features_h5.close()

## Convert to reward orientations
orient_image = 1 - orient_image

## Take the maximum angle within each 45 degree window
orient_image_round = np.zeros([norm_image.shape[0], norm_image.shape[1], 4])
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

blur_prob_scale = 2**31
blur_prob = (scipy.ndimage.gaussian_filter(prob_image, blur_sigma) * blur_prob_scale).astype(np.int32)

## Calculate the graph cut connectivity matrices

## adjacency_matrix holds the i, j distance values for each direction
## first calculate as 8 images, one for each direction

## directions start at 3 o'clock and rotate clockwise
directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
smooth_distances = np.zeros([norm_image.shape[0], norm_image.shape[1], len(directions)], dtype=np.float32)
gap_completion_distances = np.zeros([norm_image.shape[0], norm_image.shape[1], len(directions)], dtype=np.float32)

i_coords = np.array(range(norm_image.size), dtype=np.int32);
sparse_shape = (norm_image.size,norm_image.size)

n_segmentations = len(probability_thresholds) * len(smoothing_factors) * len(gap_completion_factors)
segmentations = np.zeros([norm_image.shape[0], norm_image.shape[1], n_segmentations], dtype=np.bool)
segmentation_count = 0


for di in range(len(directions)):

    direction = directions[di]

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

    current_distance = np.zeros([norm_image.shape[0], norm_image.shape[1]])

    ## Smoothing fuction on raw image
    smooth_distances[:,:,di] = 1 / (image_sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * np.power(norm_image - shifted, 2) / (image_sigma**2));

    ## Directional smoothing function on probabilities
    step_distance = np.sqrt(np.power(direction[0], 2) + np.power(direction[1], 2))
    gap_completion_distances[:,:,di] = abs(orient_image_round[:,:,di % 4]) * np.exp(-0.5 * np.power(prob_image - probability_offset, 2) / (prob_sigma**2)) / step_distance


## Loop over threshold / smoothing / gap completion settings

for probability_threshold in probability_thresholds:

    ## terminal_matrix holds +ve weights in column 1 and -ve weights in column 2

    prob_image_vector = prob_image.flatten()

    positive_vals = prob_image_vector >= probability_threshold

    terminal_matrix_prob = np.zeros([prob_image_vector.size, 2], dtype=np.float32)

    indices = np.nonzero(positive_vals)[0]
    terminal_matrix_prob[indices, 0] = 2 * (prob_image_vector[indices] - probability_threshold)

    indices = np.nonzero(np.logical_not(positive_vals))
    terminal_matrix_prob[indices, 1] = -2 * (prob_image_vector[indices] - probability_threshold)

    terminal_matrix_smooth = np.zeros([prob_image_vector.size, 2], dtype=np.float32)
    terminal_matrix_smooth[:,0] = np.sum(smooth_distances, axis=2).flatten()
    terminal_matrix_smooth[:,1] = np.sum(smooth_distances, axis=2).flatten()

    terminal_matrix_gap_completion = np.zeros([prob_image_vector.size, 2], dtype=np.float32)
    terminal_matrix_gap_completion[:,0] = np.sum(gap_completion_distances, axis=2).flatten()

    for smoothing_factor in smoothing_factors:
        for gap_completion_factor in gap_completion_factors:

            flow_graph = pymaxflow.PyGraph(terminal_matrix_prob.shape[0], smooth_distances.shape[2])

            flow_graph.add_node(terminal_matrix_prob.shape[0]);

            ## Load the terminal matrix

            terminal_matrix = terminal_matrix_prob + terminal_matrix_smooth * smoothing_factor + terminal_matrix_gap_completion * gap_completion_factor

            flow_graph.add_tweights_vectorized( i_coords, terminal_matrix[:,0].flatten(), terminal_matrix[:,1].flatten() )

            combined_distances = smooth_distances * smoothing_factor + gap_completion_distances * gap_completion_factor


            ## Load the adjacency matrix

            for di in range(len(directions)):

                direction = directions[di]

                j_coords = i_coords + 1 * direction[0] + norm_image.shape[0] * direction[1]

                valid = np.nonzero(np.logical_and(j_coords >= 0, j_coords < norm_image.size))
                rows = i_coords[valid]
                cols = j_coords[valid]

                forward_distance = np.float32(combined_distances[:,:,di].flatten()[rows])
                backward_distance = np.zeros(forward_distance.shape, dtype=np.float32)

                flow_graph.add_edge_vectorized( rows, cols, forward_distance, backward_distance )


            ## Compute the max flow

            print "Running maxflow for probability threshold {0}, smoothing factor {1}, gap completion factor {2}.".format(probability_threshold, smoothing_factor, gap_completion_factor)

            flow_graph.maxflow()

            labels = flow_graph.what_segment_vectorized()
            #labels, nlabels = mahotas.label(labels.reshape(norm_image.shape))

            labels = labels.reshape(norm_image.shape)

            ## Remove small segments
            labels = mahotas.morph.close(labels.astype(np.bool), disc)
            labels = mahotas.morph.open(labels.astype(np.bool), disc)

            #plt.figure(figsize=(15,15))
            #plt.imshow(labels, cmap=cm.gray)
            #plt.draw()

            skeleton = mahotas.thin(labels==0)
            biglabels, nlabels = mahotas.label(skeleton==0)
            biglabels = mahotas.dilate(biglabels)

            dx, dy = np.gradient(biglabels)
            boundary = np.logical_or(dx!=0, dy!=0)

            #plt.figure(figsize=(15,15))
            #plt.imshow(boundary, cmap=cm.gray)
            #plt.draw()

            ## Use blurred probabilities and watershed instead of region growing
            seeds,_ = mahotas.label(labels==1)

            ws = mahotas.cwatershed(blur_prob, seeds)
            dx, dy = np.gradient(ws)
            ws_boundary = np.logical_or(dx!=0, dy!=0)

            #plt.figure(figsize=(15,15))
            #plt.imshow(ws_boundary, cmap=cm.gray)
            #plt.draw()

            ## Identify possible extra-cellular space - distance method
            #extra_cellular = np.logical_and(mahotas.distance(labels==0) > 100, seeds == np.min(seeds))
            #extra_cellular = mahotas.morph.close(extra_cellular.astype(np.bool), disc)
            #extra_cellular = mahotas.morph.open(extra_cellular.astype(np.bool), disc)
            #extra_cellular_indices = np.nonzero(extra_cellular)

            ## Identify possible extra-cellular space - minima method
            extra_cellular = np.logical_and(mahotas.regmin(blur_prob, min_disc), seeds == np.min(seeds))
            extra_cellular_indices = np.nonzero(extra_cellular)

            extra_cellular_id = np.max(seeds)+1
            seeds[extra_cellular_indices] = extra_cellular_id

            #plt.figure(figsize=(15,15))
            #plt.imshow(seeds, cmap=cm.gray)
            #plt.draw()

            ws = mahotas.cwatershed(blur_prob, seeds)
            dx, dy = np.gradient(ws)
            ws_boundary = np.logical_or(dx!=0, dy!=0)

            ## Optional - mark the extra-cellular space as boundary
            #ws_boundary[np.nonzero(ws == extra_cellular_id)] = 1

            plt.figure(figsize=(15,15))
            plt.imshow(ws_boundary, cmap=cm.gray)
            plt.draw()

            segmentations[:,:,segmentation_count] = ws_boundary
            segmentation_count = segmentation_count + 1
