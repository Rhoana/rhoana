function h=plotsiftframe(frames,varargin)
% PLOTSIFTFRAME  Plot SIFT frame
%   H=PLOTSIFTFRAME(FRAMES) plots the SIFT frames FRAMES and returns
%   and handle H to the resulting line set. FRAMES has the same format
%   used by SIFT().
%
%   A SIFT frame is denoted by a circle, representing its support, and
%   one of its radii, representing its orientation. The support is a
%   disk with radius equal to six times the scale SIGMA of the
%   frame. If the standard parameters are used for the detector, this
%   corresponds to four times the standard deviation of the Gaussian
%   window that has been uses to estimate the orientation, which is in
%   fact equal to 1.5 times the scale SIGMA.
%
%   Option-value pairs
%
%   'Labels' []
%     Specify a cell-array of labels, one for each keypoint. These
%     will be drawn close to the keypoint centers.
%
%   'Style' ['circle']
%     Plot style: circles ('circle'),  arrows ('arrow')
%
%   This function is considerably more efficient if called once on a
%   whole set of frames as opposed to multiple times, one for each
%   frame.
%
%   See also PLOTMATCHES(), PLOTSIFTDESCRIPTOR(), PLOTSS().

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

% --------------------------------------------------------------------
%                                                  Check the arguments
% --------------------------------------------------------------------

if size(frames,1) ~= 4
  error('FRAMES should be a 4xK matrix') ;
end


putlabel = 0 ;
labels=[];
style='circle' ;

for k=1:2:length(varargin)
  switch lower(varargin{k})

    case 'style'
      switch lower(varargin{k+1})
        case 'circle'
          style = 'circle';
        case 'arrow'
          style = 'arrow' ;
        otherwise
          error(['Unknown style type ''', style, '''.']) ;
      end

    case 'labels'
      labels = varargin{k+1} ;
      putlabel = 1;

    otherwise
      error(['Unknown option ''',varargin{k},'''.']) ;
  end
end

K = size(frames,2) ;

% --------------------------------------------------------------------
%                                                          Do the work
% --------------------------------------------------------------------

hold on ;
K=size(frames,2) ;
thr=linspace(0,2*pi,40) ;

allx = nan*ones(1, 40*K+(K-1)) ;
ally = nan*ones(1, 40*K+(K-1)) ;

allxf = nan*ones(1, 3*K) ;
allyf = nan*ones(1, 3*K) ;

for k=1:K
  xc=frames(1,k)+1 ;
  yc=frames(2,k)+1 ;
  r=1.5*4*frames(3,k) ;
  th=frames(4,k) ;

  x = r*cos(thr) + xc ;
  y = r*sin(thr) + yc ;

  allx((k-1)*(41) + (1:40)) = x ;
  ally((k-1)*(41) + (1:40)) = y ;

  allxf((k-1)*3 + (1:2)) = [xc xc+r*cos(th)] ;
  allyf((k-1)*3 + (1:2)) = [yc yc+r*sin(th)] ;

  if putlabel
    x=xc+r ;
    y=yc ;
    h=text(x+2,y,sprintf('%d',labels(k))) ;
    set(h,'Color',[1 0 0]) ;
    plot(x,y,'r.') ;
  end

end

switch style
  case 'circle'
    h=line([allx nan allxf], [ally nan allyf], 'Color','g','LineWidth',3) ;
  case 'arrow'
    h=quiver(allxf(0+(1:3:3*K)),...
             allyf(0+(1:3:3*K)),...
             allxf(1+(1:3:3*K))-allxf(0+(1:3:3*K)),...
             allyf(1+(1:3:3*K))-allyf(0+(1:3:3*K)),...
             0,...
             'Color','g','LineWidth',3) ;
end
