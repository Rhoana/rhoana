% SIFT_DEMO2  Demonstrate SIFT code (2)
%   This is similar to SIFT_DEMO().
%
%   See also SIFT_DEMO().

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

I1=imreadbw('data/landscape-a.jpg') ; % I1=I1(1:2:end,:) ;
I2=imreadbw('data/landscape-b.jpg') ; % I2=I2(1:2:end,:) ;
I1c=double(imread('data/landscape-a.jpg'))/255.0 ;
I2c=double(imread('data/landscape-b.jpg'))/255.0 ;

I1=imsmooth(I1,.1) ;
I2=imsmooth(I2,.1) ;

I1=I1-min(I1(:)) ;
I1=I1/max(I1(:)) ;
I2=I2-min(I2(:)) ;
I2=I2/max(I2(:)) ;

S=3 ;

fprintf('Computing frames and descriptors.\n') ;
[frames1,descr1,gss1,dogss1] = sift( I1, 'Verbosity', 1, 'Threshold', ...
                                     0.005, 'NumLevels', S ) ;
[frames2,descr2,gss2,dogss2] = sift( I2, 'Verbosity', 1, 'Threshold', ...
                                     0.005, 'NumLevels', S ) ;

figure(11) ; clf ; plotss(dogss1) ; colormap gray ;
figure(12) ; clf ; plotss(dogss2) ; colormap gray ;
drawnow ;

figure(2) ; clf ;
tightsubplot(1,2,1) ; imagesc(I1) ; colormap gray ; axis image ;
hold on ;
h=plotsiftframe( frames1 ) ; set(h,'LineWidth',2,'Color','g') ;
h=plotsiftframe( frames1 ) ; set(h,'LineWidth',1,'Color','k') ;

tightsubplot(1,2,2) ; imagesc(I2) ; colormap gray ; axis image ;
hold on ;
h=plotsiftframe( frames2 ) ; set(h,'LineWidth',2,'Color','g') ;
h=plotsiftframe( frames2 ) ; set(h,'LineWidth',1,'Color','k') ;

fprintf('Computing matches.\n') ;
% By passing to integers we greatly enhance the matching speed (we use
% the scale factor 512 as Lowe's, but it could be greater without
% overflow)
descr1=uint8(512*descr1) ;
descr2=uint8(512*descr2) ;
tic ; 
matches=siftmatch( descr1, descr2, 3 ) ;
fprintf('Matched in %.3f s\n', toc) ;

figure(3) ; clf ;
plotmatches(I1c,I2c,frames1(1:2,:),frames2(1:2,:),matches,...
  'Stacking','v') ;
drawnow ;

% Movie
figure(4) ; set(gcf,'Position',[10 10 1024 512]) ;
figure(4) ; clf ;
tightsubplot(1,1);
imagesc(I1) ; colormap gray ; axis image ; hold on ;
h=plotsiftframe( frames1 ) ; set(h,'LineWidth',1,'Color','g') ;
h=plot(frames1(1,:),frames1(2,:),'r.') ;
MOV(1)=getframe ;

figure(4) ; clf ;
tightsubplot(1,1);
imagesc(I2) ; colormap gray ; axis image ; hold on ;
h=plotsiftframe( frames2 ) ; set(h,'LineWidth',1,'Color','g') ;
h=plot(frames2(1,:),frames2(2,:),'r.') ;
MOV(2)=getframe ;
