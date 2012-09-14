% SIFTLOCALMAX  Find local maximizers
%   SEL=SIFTLOCALMAX(F) returns the indexes of the local maximizers of
%   the Q-dimensional array F.
%
%   A local maximizer is an element whose value is greater than the
%   value of all its neighbors.  The neighbors of an element i1...iQ
%   are the subscripts j1...jQ such that iq-1 <= jq <= iq (excluding
%   i1...iQ itself).  For example, if Q=1 the neighbors of an element
%   are its predecessor and successor in the linear order; if Q=2, its
%   neighbors are the elements immediately to its north, south, west,
%   est, north-west, north-est, south-west and south-est
%   (8-neighborhood).
%
%   Points on the boundary of F are ignored (and never selected as
%   local maximizers).
%
%   SEL=SIFTLOCALMAX(F,THRESH) accepts an element as a mazimizer only
%   if it is at least THRES greater than all its neighbors.
%
%   SEL=SIFTLOCALMAX(F,THRESH,P) look for neighbors only in the first
%   P dimensions of the Q-dimensional array F. This is useful to
%   process F in ``slices''.
%
%   REMARK.  Matrices (2-array) with a singleton dimension are
%   interpreted as vectors (1-array). So for example SIFTLOCALMAX([0 1
%   0]) and SIFTLOCALMAX([0 1 0]') both return 2 as an aswer. However,
%   if [0 1 0] is to be interpreted as a 1x2 matrix, then the correct
%   answer is the empty set, as all elements are on the boundary.
%   Unfortunately MATLAB does not distinguish between vectors and
%   2-matrices with a singleton dimension.  To forece the
%   interpretation of all matrices as 2-arrays, use
%   SIFTLOCALMAX(F,TRESH,2) (but note that in this case the result is
%   always empty!).
