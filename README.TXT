Subfolders and what they are for:


Control
-------
Software to controll the overall pipeline execution flow from raw image files to result.


Alignment
---------
Anything to do with aligning and merging 2D images with each other - includes 2D stitching, and alignment of consecutive 2D images into a 3D volume.


Features
--------
Calculation of features or image properties for future use by a classifier and / or segmentation strategy.


Classifiers
-----------
Machine learning classifiers to convert pixels and or calculated image features into the desired output values (such as membrane /  not membrane).


Segmentation
------------
Strategies for converting feature / classifier output into 2D or 3D segmentations, such as watershed or ridge detection methods. Also includes related segmentation tidy-up functions such as skeletonization etc.


Unification
-----------
Strategies for joining multiple segmentation results, in 2D or in 3D, over a single image / volume or between adjicent images / volumes.


Viewers
-------
Tools and software for viewing volumes or results at any stage of the pipeline.


Evaluation
----------
Different evaluation metrics to access the quality of the reconstruction.