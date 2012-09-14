% SIFTREFINEMX  Subpixel localization, thresholding and on-edge test
%   Q = SIFTREFINEMX(P, OCTAVE, SMIN) refines, thresholds and performs
%   the on-edge test for the SIFT frames P extracted from the DOG
%   octave OCTAVE with parameter SMIN (see GAUSSIANSS()).
%
%   Q = SIFTREFINEMX(P, OCTAVE, SMIN, THRESH, R) specifies custom
%   values for the local maximum strength threshold THRESH and the
%   local maximum peakedeness threshold R.
%
%   OCTAVE is an octave of the Difference Of Gaussian scale space. P
%   is a 3xK matrix specifying the indexes (X,Y,S) of the points of
%   extremum of the octave OCTAVE. The spatial indexes X,Y are integer
%   with base zero. The scale index S is integer with base SMIN and
%   represents a scale sublevel in the specified octave.
%
%   The function returns a matrix Q containing the refined keypoints.
%   The matrix has the same format as P, except that the indexes are
%   now fractional. The function drops the points that do not satisfy
%   the strength and peakedness tests.
%
%   See also SIFT().

% AUTORIGHTS
% Copyright (c) 2006 The Regents of the University of California.
% All Rights Reserved.
% 
% Created by Andrea Vedaldi
% UCLA Vision Lab - Department of Computer Science
% 
% Permission to use, copy, modify, and distribute this software and its
% documentation for educational, research and non-profit purposes,
% without fee, and without a written agreement is hereby granted,
% provided that the above copyright notice, this paragraph and the
% following three paragraphs appear in all copies.
% 
% This software program and documentation are copyrighted by The Regents
% of the University of California. The software program and
% documentation are supplied "as is", without any accompanying services
% from The Regents. The Regents does not warrant that the operation of
% the program will be uninterrupted or error-free. The end-user
% understands that the program was developed for research purposes and
% is advised not to rely exclusively on the program for any reason.
% 
% This software embodies a method for which the following patent has
% been issued: "Method and apparatus for identifying scale invariant
% features in an image and use of same for locating an object in an
% image," David G. Lowe, US Patent 6,711,293 (March 23,
% 2004). Provisional application filed March 8, 1999. Asignee: The
% University of British Columbia.
% 
% IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
% FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
% INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
% ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
% ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
% CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
% BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
% MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

