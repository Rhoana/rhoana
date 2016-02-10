# Run the full pipeline on the S1 AC3 dataset.
#
# The AC3 ("affine crop #3") dataset is a subvolume of the data captured for
# Kasthuri, et al. 2015 [\cite{TODO}], consisting of 255 slices of size
# 2048x2048, with 6nm pixels and approximately 30nm slice thickness.  Ground
# truth was produced by Daniel Berger.

from rhoana.reproducibility import fetch
from rhoana.io import ingest_zip

from rhoana.backends.compute import LocalRunner
from rhoana.backends.storage import HDF5Storage
# from rhoana.membranes import ClassifierCNN
# from rhoana.segmentation.overseg import overseg_by_watershed
# from rhoana.segmentation import join_by_threshold
# from rhoana.blocking import dice, extract_block, embed_block, neighbor_pairs
# from rhoana.blocking import stable_marriage_overlap
# from rhoana.relabel import combine_remap_tables
# from rhoana.reproducibility.evaluate import F_rand_and_VI
# 
import os.path

import logging
logging.basicConfig(level=logging.DEBUG)


# TODO: configurable
WORK_DIR = 'datasets/AC3'
CACHE_DIR = os.path.join(WORK_DIR, 'cache')
SEGMENT_JOIN_THRESHOLD = 0.5

if __name__ == '__main__':
    # set up directories if needed
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # fetch and cache data
    data_zip = fetch('AC3.zip', CACHE_DIR)
    ground_truth_zip = fetch('AC3-gt.zip', CACHE_DIR)
    membrane_classifier_file = fetch('AC3-membrane-classifier.pkl', CACHE_DIR)

    # set up data storage
    storage = HDF5Storage(os.path.join(WORK_DIR, 'data'), truncate=True)

    # set up job runner
    runner = LocalRunner(storage, num_cpus=1)

    # import raw images
    raw_data = ingest_zip(data_zip, storage)
    print("Ingested raw data volume: {} slices, {}x{} pixels, {} channel(s)"
          .format(*(raw_data.shape)))

    # classify membranes
    classifier = ClassifierCNN(membrane_classifier_file)
    for img_name, idx in raw_data.slice_indices():
        assert img_name == 'raw', "Data doesn't appear to be raw images"
        runner(classifier,
               [("raw", idx)],  # job input
               [("membrane_probabilities", idx)]  # job output
               )

    # oversegment
    for _, idx in storage.get_keys("membrane_probabilities"):
        runner(overseg_by_watershed,
               [("membrane_probabilities", idx)],
               [("oversegmentation", idx)])

    # dice
    block_coords = dice(raw_data.depth, raw_data.width, raw_data.height,
                        2, 2, 2)

    for block_coord in block_coords:
        runner(extract_block,
               [("oversegmentation", block_coord)],
               [("overseg_block", block_coord)])
        runner(extract_block,
               [("membrane_probabilities", block_coord)],
               [("probability_block", block_coord)])

    # join oversegmentations
    for block_coord in block_coords:
        runner(join_by_threshold,
               [("overseg_block", block_coord), SEGMENT_JOIN_THRESHOLD],
               [("segmentation_block", block_coord)])

    # join overlapping regions
    overlaps = neighbor_pairs(block_coords)
    for block_coord_1, block_coord_2 in overlaps:
        runner(stable_marriage_overlap,
               [("segmentation_block", block_coord_1), ("segmentation_block", block_coord_2)],
               # remap_tables are not volumetric data, so are stored with string keys
               [("remap_table", str((block_coord_1, block_coord_2)))])

    # global remap table
    runner(combine_remap_tables,
           [("remap_table", pair) for pair in overlaps],
           [("global_remap", "global")])

    # relabel individual blocks
    for block_coord in block_coords:
        runner(join_by_threshold,
               [("segmentation_block", block_coord), ("global_remap", "global")],
               [("relabeled_block", block_coord)])

    # reassemble
    for block_coord in block_coords:
        runner(embed_block,
               [("relabeled_block", block_coord)],
               [("reconstruction", block_coord)])

    # validate
    # TODO: what's the right way to do this?
    ground_truth = ingest_zip(ground_truth_zip, storage)
    runner(F_rand_and_VI,
           [("reconstruction", None)],
           [("evaluation", "")])

    # export
    # TODO choose what to export, print
