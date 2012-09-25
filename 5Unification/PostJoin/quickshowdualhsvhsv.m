function quickshowdualhsvhsv (src3, img3)
%Display a 3D image as a movie of 2D images
nsegments = max(img3(:));
reps = ceil(nsegments/length(lines));
segment_colours = [0,0,0;repmat(hsv,reps,1)];

srcmi = min(src3(:));
srcma = max(src3(:));
if srcmi == srcma
    srcma = srcmi + 1;
end

mi = min(img3(:));
ma = max(img3(:));
if mi == ma
    ma = mi + 1;
end

scrsz = get(0,'ScreenSize');
figure(1);
set(1,'Position',[8,70, scrsz(3)/2, scrsz(4)-148]);
figure(2);
set(2,'Position',[scrsz(3)/2+8,70, scrsz(3)/2-16, scrsz(4)-148]);

for z = 1:min(size(src3,3),size(img3,3))
    set(0,'CurrentFigure',1)
    colormap(segment_colours);
    imagesc(src3(:,:,z), [srcmi srcma]);
    axis image off;
    set(0,'CurrentFigure',2)
    colormap(segment_colours);
    imagesc(img3(:,:,z), [mi ma]);
    axis image off;
    pause(0.5);
end
    