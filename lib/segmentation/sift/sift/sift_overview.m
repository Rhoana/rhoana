% SIFT_OVERVIEW  Scale-Invariant Feature Transfrom
%
%  This is a MATLAB/C implementation of the SIFT detector and
%  descriptor [1]. You can:
%
%  * Use SIFT() to detect the SIFT frames (keypoints) of a given image
%    and compute their descriptors. Then you can use SIFTMATCH() to
%    match the descriptors.
%
%  * Use PLOTSS(), PLOTSIFTDESCRIPTOR(), PLOTSIFTFRAME(),
%    PLOTMATCHES() to visualize the results.
%
%  As SIFT is implemented by several reusable M and MEX files, you can
%  also run portions of the algorithm. Specifically, you can:
%
%  * Use SIFTDESCRIPTOR() to compute the SIFT descriptor from a list
%    of frames and a scale space or plain image.
%
%  * Use GAUSSIANSS() and DIFFSS() to compute the Gaussian and DOG
%    scale spaces.
%
%  * Use SIFTLOCALMAX(), SIFTREFINEMX(), SIFTORMX() to manually
%    extract the SIFT frames from the DOG scale space. More in
%    general, you can use SIFTLOCALMAX() to find maximizers of any
%    multi-dimensional arrays.
%
%  REFERENCES
%
%  [1] D. G. Lowe, "Distinctive image features from scale-invariant
%      keypoints," IJCV, vol. 2, no. 60, pp. 91 110, 2004.
%
%  See also PDF:SIFT.INTRODUCTION.

