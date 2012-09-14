function [frames,descriptors] = siftread(file) 
% SIFTREAD Read Lowe's SIFT implementation data files
%   [FRAMES, DESCRIPTORS] = READSIFT(FILE) reads the frames and the
%   descriptors from the specified file. The function reads files
%   produced by Lowe's SIFT implementation.
%     
%   FRAMES and DESCRIPTORS have the same format used by SIFT(). 
%
%   REMARK. Lowe's and our implementations use a silightly different
%   convention to store the orientation of the frame. When the file
%   is read, the orientation is changed to match our convention.
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

verbosity=0 ;

g = fopen(file, 'r');
if g == -1
    error(['Could not open file ''', file, '''.']) ;
end
[header, count] = fscanf(g, '%d', [1 2]) ;
if count ~= 2
    error('Invalid keypoint file header.');
end
K = header(1) ; 
DL = header(2) ;

if(verbosity > 0)
	fprintf('%d keypoints, %d descriptor length.\n', K, DL) ;
end

%creates two output matrices
P = zeros(4,K) ;
L = zeros(DL,K) ;

%parse tmp.key
for k = 1:K

	% Record format: i,j,s,th
	[record, count] = fscanf(g, '%f', [1 4]) ; 
	if count ~= 4
		error(...
			sprintf('Invalid keypoint file (parsing keypoint %d)',k) );
	end
	P(:,k) = record(:) ;
	
	% Record format: descriptor
	[record, count] = fscanf(g, '%d', [1 DL]) ;
	if count ~= DL
		error(...
			sprintf('Invalid keypoint file (parsing keypoint %d)',k) );
	end
	L(:,k) = record(:) ;
	
end
fclose(g) ;

L=double(L) ;
P(1:2,:)=flipud(P(1:2,:)) ; % i,j -> x,y

frames=[ P(1:2,:) ; P(3,:) ; -P(4,:) ] ;
descriptors = L ;
