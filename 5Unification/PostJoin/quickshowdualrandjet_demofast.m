function quickshowdualrandjet_demofast (src3, img3)
%Display a 3D image as a movie of 2D images
%nsegments = max(img3(:));
%reps = ceil(single(nsegments)/single(length(lines)));
%segment_colours = [0,0,0;repmat(hsv,reps,1)];

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
colormap(gray);
figure(2);
set(gcf,'MenuBar', 'none');
set(gcf,'ToolBar', 'none');
set(2,'Position',[scrsz(3)/2,70, scrsz(3)/2-48, scrsz(4)-148]);
set(gca,'Position',[0 0 1 1]);

stream=RandStream('mt19937ar', 'Seed', 0);

jetcolours = [jet(63); hsv(63)];
[sorted jetorder] = sort(rand(stream,length(jetcolours),1));
nsegments = max(img3(:));
reps = ceil(single(nsegments)/single(length(lines)));
jetcolours = [0,0,0;repmat(jetcolours(jetorder,:),reps,1)];
colormap(jetcolours);

for z = 1:min(size(src3,3),size(img3,3))
    set(0,'CurrentFigure',1)
    imagesc(src3(:,:,z), [srcmi srcma]);
    axis image off;
    set(0,'CurrentFigure',2)
    imagesc(img3(:,:,z), [mi ma]);
    axis image off;
    pause(0.1);
end

    