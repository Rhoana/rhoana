% SIFTDESCRIPTOR  Compute SIFT descriptors
%   DESCR = SIFTDESCRIPTOR(G, P, SIGMA0, S, MINS) computes the SIFT
%   descriptors DESCR of the SIFT frames P defined on the octave G of
%   a Gaussian scale space. SIGMA0, S and MINS are the the parameter
%   of the scale space as explained in PDF:SIFT.USER.SS. P has one
%   column per frame, specifiying the center X1,X2, the scal index s
%   and the orientation THETA of the frame.
%
%   As the function operates on a single octave, in order to process
%   features spanning several octaves, one must group them and call
%   SIFTDESCRIPTOR() once per each octave.
%    
%   DESCR = SIFTDESCRIPTOR(I, P, SIGMA0) operates on a plain image
%   I. The image I is assumed to be pre-smoothed at scale SIGMA0 and P
%   is a matrix with a column per frame, specifying the center X1,X2
%   and the orientation THETA (but NOT the scale).
%
%   REMARK. The scale parameter s in P is the scale index, NOT the
%   scale coordinate (see the PDF doc. for a discussion).
%
%   Other parameters can be specfied as option-value paris. These
%   are
%
%   'Magnif' [3.0]
%      Frame magnification factor. Each spatial bin of the SIFT
%      histogram has an exentsion equal to magnif * sigma, where
%      magnif is the frame magnification factor and sigma is the scale
%      of the frame.
%
%   'NumSpatialBins' [4]
%      This parameter specifies the number of spatial bins in each
%      spatial direction x1 and x2. It must be a positive and even
%      number.
%
%   'NumOrientBins' [8]
%      This parameter specifies the number of orietnation bins. It
%      must be a positive number.
%
%   See also SIFT(), GAUSSIANSS(), DIFFSS(), SIFTLOCALMAX(), 
%            PDF:SIFT.USER.DESCRIPTOR.

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
