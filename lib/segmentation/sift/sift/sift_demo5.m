% SIFT_DEMO5  Demonstrate SIFT code (5)
%   Finding eggs.

I=imreadbw('data/nest.png') ;

[f,d,gss,dogss] = sift(I,'verbosity',1,'boundarypoint',0,'threshold',.0282,'firstoctave',-1,'edgethreshold',0) ;
d = uint8(512*d) ;

figure(1) ; clf ; colormap gray ;
imagesc(I) ; hold on ; axis equal ;
h=plotsiftframe(f) ;
set(h,'LineWidth',3) ;

figure(2); clf ; plotss(dogss) ; colormap gray;
