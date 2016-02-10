Pipeline flow.  Datasets are in **bold**.  Note that some datasets have multiple overlapping pieces.
- Create a ``Storage`` object for raw images, workspace, and results.  This is where ``Datasets`` live.
- Create a ``Runner`` object to handle jobs and dependencies.
- Ingest (via ``io`` module) raw images into a **Raw** data volume.
- For each plane in the **Raw** volume:
    - Compute & store **Membrane** probabilities (this may require multiple slices of input).
    - Run pre-segmentation (e.g., watershed) on the current **Membrane** probabilities, and store as **Oversegmentations**.
- Dice the **Oversegmentations** into **Oversegmentation Cubes**
- For each **Oversegmentation Cube**:
     - Run Agglomeration or Fusion, store as **Segmentation Cubes**
- For each pair of adjacenet **Segmentation Cubes**:
     - Run pairwise matching (e.g., Stable Marriage) and store a local **Matching Remap**.
     - (the **Matching Remap** may need to be applied to **Segmentation Cubes** before each sequent pairwise matching.)
- Combing the **Matching Remaps** into a single **Global Remap**
- For each **Segmentation Cube**
     - Remap the **Segmentation Cube** through the **Global Remap**
     - Write the result to the **Global Segmentation**


Datasets are referenced in Storage by name (and extent, maybe? in the case of things like cubes).  They provide slicing operators (returning numpy arrays or similar), as well as slice assignment.  Efficient in-place computation may be supported, but this will depend on the backend.

The Runner uses dataset names to determine dependencies.  For example, to create a task/job in the runner to extract a block of segmentations, one would call:
```
runner(extract_block,
       input=[("Oversegmentations", block_coord)],
       output=[("OversegmentationBlock{}".format(block_coord), block_coord)])
```
In this case, the ``extract_block`` job will not be run until any jobs that output to ``Oversegmentations`` have completed.  Note that  ``block_coords`` in the output may not be anchored at ``(0,0,0)``, though this may change.

**TODO**: how to handle this correctly?  Associate an origin with each dataset?  Is slicing relative to origin or global position?

**TODO**: In the future, dependencies should also look at coordinates, probably using logic in the ``extract_block`` object.

