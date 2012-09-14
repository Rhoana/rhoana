% SIFT_DEMO6  Demonstrate SIFT code (6)
%   Using custom keypoints.

I=imreadbw('data/nest.png') ;

% first run the complete pipeline
[f,d] = sift(I,'verbosity',0,'threshold',.0282) ;

f
% now select a specific keypoint
f = f(:,end) ;
d = d(:,end) ;

%
% Manual computation of a descriptor
%

% pre-smooth image at the right level
Is = imsmooth(I,f(3)) ;

% compute descriptor
dp = siftdescriptor(Is,f([1 2 4]),f(3)) ;

% the same computation can be carried also by downsampling
Is = Is(1:2:end,1:2:end) ;
dpp = siftdescriptor(Is,diag([.5 .5 1])*f([1 2 4]),f(3)/2) ;

figure(1) ; clf ; colormap gray ;
subplot(1,2,1) ;
imagesc(I) ; hold on ; axis equal ;
h=plotsiftdescriptor(d,f) ; set(h,'linewidth',2,'color','b')
h=plotsiftdescriptor(dp,f) ; set(h,'linewidth',2,'color','y')
h=plotsiftdescriptor(dpp,f) ; set(h,'linewidth',2,'color','k')
h=plotsiftframe(f) ;

subplot(1,2,2) ;
hold on ;
plot([d dp dpp]) ;
legend('auto','manual','manual with ds') ;


