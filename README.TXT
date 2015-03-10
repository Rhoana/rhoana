NOTE: The Rhoana pipeline is still under development, and should not be considered stable.

Rhoana - Dense Automated Neuron Annotation Pipeline

Prerequisites:
numpy          http://numpy.org
scipy          http://scipy.org
h5py           http://www.h5py.org/
mahotas        http://luispedro.org/software/mahotas
OpenCV         http://opencv.org/
pymaxflow      https://github.com/Rhoana/pymaxflow
fast64counter  https://github.com/Rhoana/fast64counter
CPLEX          http://www.ibm.com/software/integration/optimization/cplex-optimizer/


The Rhoana pipeline operates in the following stages:
Classify Membranes
Segmentation
Block dicing
Window Fusion
Pairwise Matching
Local and Global Remapping

A simple driver program is in Control/driver.py.  It takes as input a
file containing a list of images to process.  These should be aligned
EM sections.

ClassifyMembranes/classify_image takes three arguments:
  - image file
  - classifier file (an example is in ClassifyMembranes/GB_classifier.txt)
  - output HDF5
The HDF5 output will contain a single dataset, "probabilities", which
are the per-pixel membrane probabilities.

Segment/segment.py takes two arguments:
  - probabilities HDF5
  - output segmentations HDF5
Output will contain two datasets, "segmentations" and
"probabilities".  The first is of size IxJxN, with I,J the image
dimensions and N the number of generated segmentations (at various
scale and smoothness, N = 30 in the current implementation). The
"probabilities" dataset is just copied from the input.

Control/dice_block.py takes a number of arguments:
- imin, jmin, imax, jmax - the IJ coordinates of the block
- output.hdf5
- K segmentation HDF5 files
This will cut out a block as:
     np.concat([im[imin:imax, jmin:jmax, :] for im in segs[K]], 4)
(and a similar block for the per-pixel probabilities)
It will produce two datasets, "segmentations" and "probabilities".  Segmentation

WindowFusion/window_fusion_cpx.py takes two arguments:
- input block.hdf5
- output fusedblock.hdf5
This will run window fusion to reduce the IxJxNxK block to a labeled
IxJxK block.  Two datasets are produced, "labels" and "probabilities".

PairwiseMatching/pairwise_match.py takes 6 arguments
- two input fused blocks
- the direction they overlap (X = 1, Y = 2, Z = 3)  # this may be inaccurate, currently
- the number of pixels they overlap
- two output HDF5 fused blocks
Pairwise matching produces "labels", "probabilities", and "merges"
datasets.  The first block should always be closer to 0,0,0.  The
usual method is to run it first for all X-even blocks matching to
their X+1 neighbor, then all X-odd blocks matching to their X+1
neighbor, then do the same for Y, then Z.  After Pairwise Matching,
overlapping regions should be consistent.  "merges" is Lx2, with each
row indicating that two labels should be merged in the final result.

(There is a similar, program pairwise_match_region_growing.py, that
uses region growing in the probability maps for overlapping regions.)

Relabelabeling/concatenate_joins.py takes multiple matches blocks and
extracts their merges, and Relabelabeling/create_global_map.py
processes the full list of merges to create the final remap function.
Relabeling/remap_block.py takes this global remap and a single block,
and produces the remapped block.

Relabeling/extract_label_plane.py takes the following arguments:
- the output hdf5 path
- its IxJ size (same as the original image)
- a Z offset for the plane within the input blocks
- a set of (ibase, jbase, input block HDF5)
Extract Label Plane performs rougly the following action:
  for ibase, jbase, infile in args:
     input_data = infile['labels'][:, :, Z]
     output_labels[ibase:ibase+input_data.shape[0],
                   jbase:jbase+input_data.shape[1]] = intput_data
