% SIFT_DEMO3  Demonstrate SIFT code (3)
%   This compares our and D. Lowe's implementation. Not perfect,
%   but close.

I=imreadbw('data/box.pgm') ;
[f,d]   = sift(I,'verbosity',1,'boundarypoint',0) ; d = uint8(512*d) ;
[lf,ld] = siftread('data/box.sift') ; ld = uint8(ld) ;
matches = siftmatch(d,ld) ; 

figure(1) ; clf ; colormap gray ;
subplot(2,1,1) ;
imagesc(I) ; hold on ;
plotsiftframe(f,'style','arrow') ;
title('This implementation') ;

subplot(2,1,2) ;
imagesc(I) ; hold on ;
plotsiftframe(lf,'style','arrow') ;
title('D. Lowe''s implementation') ;

figure(2) ; clf ; colormap gray; 
plotmatches(I,I,f,lf,matches) ;
title('This implementation and Lowe''s matches') ;

figure(3) ; clf ; colormap gray ;
imagesc(I) ; hold on ;
lh = plotsiftframe(lf,'style','arrow') ; hold on ;
h  = plotsiftframe(f,'style','arrow');
set(lh,'LineWidth',2) ;
set(h,'Color','r','LineWidth',1); 
title('This implementation and Lowe''s') ;

% difference in th, ordered by scale
df = f(:,matches(1,:))-lf(:,matches(2,:)) ;
df(4,:)=mod(df(4,:)+pi,2*pi)-pi ;
%df=abs(df) ;

% keep only inlier matches
sel=find(sum(df.^2)<1) ;
df=df(:,sel) ;

figure(4) ; clf ;  
subplot(2,2,1) ; hold on ; 
plot(df(4,:)) ; plot(2*pi/36*ones(size(df(4,:)))) ;
title('Delta orientations') ;

subplot(2,2,2) ; hold on ;
plot(df(1,:)) ; plot(ones(size(df(1,:)))) ;
title('Delta x') ;

subplot(2,2,3) ; hold on ;
plot(df(2,:)) ; plot(ones(size(df(1,:)))) ;
title('Delta y') ;

subplot(2,2,4) ; 
plot(df(3,:)) ; 
title('Delta sigma') ;

figure(5) ; clf ;
K=9 ;
for k=1:K
  tightsubplot(K,k) ;
  hold on ;
  h=plotsiftdescriptor(d(:,matches(1,k))) ;
  lh=plotsiftdescriptor(ld(:,matches(2,k))) ;
  set(h,'LineWidth',2) ;
  set(lh,'Color','r') ;
  axis tight;axis square;axis off;
end