function plotss(ss,field)
% PLOTSS  Plot scale space
%   PLOTSS(SS) plots all octaves of the scale space SS.
%
%   See also GAUSSIANSS(), DIFFSS().

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

if nargin > 2
    error('Too many arguments.') ;
end

omin = ss.omin ;
smin = ss.smin ;
nlevels = ss.smax-ss.smin+1 ;

for oi=1:ss.O
  for si=1:nlevels
    tightsubplot(nlevels, ss.O, nlevels*(oi-1)+si) ;
    s = si-1 + smin ;
    o = oi-1 + omin ;
    sigma = ss.sigma0 * 2^(s/ss.S + o) ;
    F=squeeze(ss.octave{oi}(:,:,si)) ;
    [M,N]=size(F) ;
    imagesc(squeeze(ss.octave{oi}(:,:,si))) ; axis image ; axis off ;
    h=text(M/10,N/20,sprintf('(o,s)=(%d,%d), sigma=%f',o,s,sigma)) ;
    set(h,'BackgroundColor','w','Color','k') ;
  end
end


