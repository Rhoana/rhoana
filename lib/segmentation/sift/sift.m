function [frames,descriptors,gss,dogss]=sift(I,varargin)
% SIFT Extract SIFT frames (keypoints) and descriptors
%   [FRAMES,DESCRIPTORS]=SIFT(I) extracts the SIFT frames FRAMES and
%   the corresponding descriptors DESCRIPTORS from the image I.  
%
%   The image I is assumed to be gray-scale, in DOBULE storage
%   class and range normalised in [0,1].
%
%   FRAMES is a 4xK matrix, storing one frame per column, and has the
%   following format:
%
%     FRAMES(1:2,k)  center (X,Y) of the frame k,
%     FRAMES(3,k)    scale SIGMA of the frame k,
%     FRAMES(4,k)    orientation THETA of the frame k.
%
%   The coordinates (X,Y) of the frame center are relative to the
%   upper-left corner of the image plane, which is assigned
%   coordinates (0,0).
%
%   DESCRIPTORS stores one descriptor per column. Usually this matrix
%   has dimension 128xK, but the number of rows can change with the
%   parameteres of the algorithm.
%
%   [FRAMES,DESCRIPTORS,GSS,DOGSS]=SIFT(I) returns the Gaussian and
%   Difference of Gaussians scale spaces as well.
%
%   The function accepts several option-value pairs, specifying
%   parameters of the algorithm (the default values are chosen to
%   emulate Lowe's original implementation):
%
%   'Verbosity' [0]
%     Verbosity level (0=quiet)
%
%   'BoundaryPoint' [1]
%     Remove points whose descriptor intersects the boundary.
%
%   'NumOctaves' [1,2,...]
%     Number of octaves of the Gaussian scale space
%
%   'FirstOctave' [..,-1,0,1,...]
%     Index of the first octave. Fetting the parameter to -1 has the
%     effect of doubling the image before computing the scale space.
%
%   'NumLevels' [1,2,...]
%     Number of scale levels within each octave.
%
%   'Sigma0' [pixels]
%     Smoothing of the level 0 of octave 0 of the scale space.
%     (Note that Lowe's 1.6 value refers to the level -1 of
%     octave 0.)
%
%   'SigmaN' [pixels]
%     Nominal smoothing of the input image. Typically set to 0.5.
%
%   'Threshold'
%     Threshold used to eliminate weak keypoints. Typical values for
%     intensity images in the range [0,1] are around 0.01. Smaller
%     values mean more keypoints.
%
%   'EdgeThreshold'
%     Threshold used to eliminate keypoints on edegs. Typical values
%     are round 10. Bigger values mean more keypoints.
%
%   'Magnif'
%     Frame magnification when computing descriptor (see
%     SIFTDESCRIPTOR()).
%
%   'NumSpatialBins'
%     Number of spatial bins of the SIFT descritpor (see SIFTDESCRIPTOR()).
%
%   'NumOrientbins'
%     Number of orientation bins of the SIFT descriptor( see
%     SIFTDESCRIPTOR()).
%
%   See also GAUSSIANSS(), DIFFSS(), PLOTSIFTFRAME(), PLOTSIFTDESCRIPTOR(),
%            SIFTDESCRIPTOR(), SIFTMATCH().
%

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

% TODO: fix convention for x1,x2 (matlab or C?)
%       fix convention for descriptors (our or Lowe's?)

% --------------------------------------------------------------------
%                                                  Check the arguments
% --------------------------------------------------------------------
if(nargin < 1)
	error('At least one argument is required.') ;
end

[M,N,C] = size(I) ;

% Lowe's choices
S=3 ;
omin=-1 ;
O=floor(log2(min(M,N)))-omin-3 ; % Up to 8x8 images
sigma0=1.6*2^(1/S) ;
sigman=0.5 ;
thresh = 0.04 / S / 2 ;
r = 10 ;

NBP = 4 ;
NBO = 8 ;
magnif = 3.0 ;

% Parese input
compute_descriptor = 0 ;
discard_boundary_points = 1 ;
verb = 0 ;

for k=1:2:length(varargin)
	switch lower(varargin{k})
    
    case 'numoctaves'
      O = varargin{k+1} ;
      
    case 'firstoctave'
      omin = varargin{k+1} ;
      
    case 'numlevels'
      S = varargin{k+1} ;

    case 'sigma0'
      sigma0 = varargin{k+1} ;
      
    case 'sigman'
      sigmaN = varargin{k+1} ;
      
    case 'threshold'
      thresh = varargin{k+1} ;
      
    case 'edgethreshold'
      r = varargin{k+1} ;
      
    case 'boundarypoint'
     discard_boundary_points = varargin{k+1} ;
     
    case 'numspatialbins'
      NBP = varargin{k+1} ;
      
    case 'numorientbins'
      NBO = varargin{k+1} ;
      
    case 'magnif'
      maginf = varargin{k+1} ;
                                    
    case 'verbosity' 
     verb = varargin{k+1} ;

    otherwise
      error(['Unknown parameter ' varargin{k} '.']) ;
  end
end

% Arguments sanity check
if C > 1
  error('I should be a grayscale image') ;
end



% --------------------------------------------------------------------
%                                                           Parameters
% --------------------------------------------------------------------

frames = [] ;
descriptors = [] ;

% --------------------------------------------------------------------
%                                         SIFT Detector and Descriptor
% --------------------------------------------------------------------

% Compute scale spaces
if verb > 0, fprintf('SIFT: computing scale space...') ; tic ; end

gss = gaussianss(I,sigman,O,S,omin,-1,S+1,sigma0) ;

if verb > 0, fprintf('(%.3f s gss; ',toc) ; tic ; end 

dogss = diffss(gss) ;

if verb > 0, fprintf('%.3f s dogss) done\n',toc) ; end
if verb > 0
	fprintf('SIFT scale space parameters [PropertyName in brackets]\n');
	fprintf('  sigman [SigmaN]        : %f\n', sigman) ;
	fprintf('  sigma0 [Sigma0]        : %f\n', dogss.sigma0) ;
	fprintf('       O [NumOctaves]    : %d\n', dogss.O) ;
	fprintf('       S [NumLevels]     : %d\n', dogss.S) ;
	fprintf('    omin [FirstOctave]   : %d\n', dogss.omin) ;
	fprintf('    smin                 : %d\n', dogss.smin) ;
	fprintf('    smax                 : %d\n', dogss.smax) ;
	fprintf('SIFT detector parameters\n')
	fprintf('  thersh [Threshold]     : %e\n', thresh) ;
  fprintf('       r [EdgeThreshold] : %.3f\n', r) ;
  fprintf('SIFT descriptor parameters\n')
  fprintf('  magnif [Magnif]        : %.3f\n', magnif) ;
  fprintf('     NBP [NumSpatialBins]: %d\n', NBP) ;
  fprintf('     NBO [NumOrientBins] : %d\n', NBO) ;
end


for o=1:gss.O
	if verb > 0
		fprintf('SIFT: processing octave %d\n', o-1+omin) ;
                tic ;
	end
	
	% Local maxima of the DOG octave
	% The 80% tricks discards early very weak points before refinement.
	idx = siftlocalmax(  dogss.octave{o}, 0.8*thresh  ) ;
	idx = [idx , siftlocalmax( - dogss.octave{o}, 0.8*thresh)] ;  
  
	K=length(idx) ; 
	[i,j,s] = ind2sub( size( dogss.octave{o} ), idx ) ;
	y=i-1 ;
	x=j-1 ;
	s=s-1+dogss.smin ;
  oframes = [x(:)';y(:)';s(:)'] ;
	
	if verb > 0
    fprintf('SIFT: %d initial points (%.3f s)\n', ...
      size(oframes, 2), toc) ;
    tic ;
	end
	
	% Remove points too close to the boundary
	if discard_boundary_points
    % radius = maginf * sigma * NBP / 2
    % sigma = sigma0 * 2^s/S
    
    rad = magnif * gss.sigma0 * 2.^(oframes(3,:)/gss.S) * NBP / 2 ;
    sel=find(...
      oframes(1,:)-rad >= 1                     & ...
      oframes(1,:)+rad <= size(gss.octave{o},2) & ...
      oframes(2,:)-rad >= 1                     & ...
      oframes(2,:)+rad <= size(gss.octave{o},1)     ) ;
    oframes=oframes(:,sel) ;
		
    if verb > 0
			fprintf('SIFT: %d away from boundary\n', size(oframes,2)) ;
      tic ;
    end
	end
		
	% Refine the location, threshold strength and remove points on edges
	oframes = siftrefinemx(...
		oframes, ...
		dogss.octave{o}, ...
		dogss.smin, ...
		thresh, ...
		r) ;
  
	if verb > 0
		fprintf('SIFT: %d refined (%.3f s)\n', ...
            size(oframes,2),toc) ;
    tic ;
  end
  
	% Compute the oritentations
	oframes = siftormx(...
		oframes, ...
		gss.octave{o}, ...
		gss.S, ...
		gss.smin, ...
		gss.sigma0 ) ;
			
  % Store frames
	x     = 2^(o-1+gss.omin) * oframes(1,:) ;
	y     = 2^(o-1+gss.omin) * oframes(2,:) ;
	sigma = 2^(o-1+gss.omin) * gss.sigma0 * 2.^(oframes(3,:)/gss.S) ;		
	frames = [frames, [x(:)' ; y(:)' ; sigma(:)' ; oframes(4,:)] ] ;

	
	% Descriptors
	if nargout > 1
		if verb > 0
      fprintf('SIFT: computing descriptors...') ;
      tic ;
    end
		
		sh = siftdescriptor(...
			gss.octave{o}, ...
      oframes, ...
      gss.sigma0, ...
      gss.S, ...
      gss.smin, ...
      'Magnif', magnif, ...
      'NumSpatialBins', NBP, ...
      'NumOrientBins', NBO) ;
    
    descriptors = [descriptors, sh] ;
    
    if verb > 0, fprintf('done (%.3f s)\n',toc) ; end
  end
end
