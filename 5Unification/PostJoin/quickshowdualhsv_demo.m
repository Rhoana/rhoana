function quickshowdualhsv_demo (src3, img3)
%Display a 3D image as a movie of 2D images
nsegments = max(img3(:));
reps = ceil(single(nsegments)/128);
segment_colours = [0,0,0;repmat([hsv(64); jet(64)],reps,1)];

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
set(gcf,'MenuBar', 'none');
set(gcf,'ToolBar', 'none');
set(1,'Position',[38,70, scrsz(3)/2-48, scrsz(4)-148]);
set(gca,'Position',[0 0 1 1]);
figure(2);
set(gcf,'MenuBar', 'none');
set(gcf,'ToolBar', 'none');
set(2,'Position',[scrsz(3)/2,70, scrsz(3)/2-48, scrsz(4)-148]);
set(gca,'Position',[0 0 1 1]);

for z = 1:min(size(src3,3),size(img3,3))
    set(0,'CurrentFigure',1)
    colormap(gray);
    imagesc(src3(:,:,z), [srcmi srcma]);
    axis image off;
    set(0,'CurrentFigure',2)
    colormap(segment_colours);
    imagesc(img3(:,:,z), [mi ma]);
    axis image off;
    pause(0.2);
    if z == 1
        pause(1);
    end
end

    