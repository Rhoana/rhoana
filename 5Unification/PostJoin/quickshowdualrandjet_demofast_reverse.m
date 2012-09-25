function quickshowdualrandjet_demofast (src3, img3, reverse)
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
set(1,'Position',[38,70, scrsz(3)/2-128, scrsz(4)-228]);
set(gca,'Position',[0 0 1 1]);
colormap(gray);
figure(2);
set(gcf,'MenuBar', 'none');
set(gcf,'ToolBar', 'none');
set(2,'Position',[scrsz(3)/2-80,70, scrsz(3)/2-128, scrsz(4)-228]);
set(gca,'Position',[0 0 1 1]);

stream=RandStream('mt19937ar', 'Seed', 11);
jetcolours = [jet(128); hsv(128).*0.85];
[sorted jetorder] = sort(rand(stream,length(jetcolours),1));
nsegments = max(img3(:));
reps = ceil(single(nsegments)/single(length(jetcolours)));
jetcolours = [0,0,0;repmat(jetcolours(jetorder,:),reps,1)];
%jetcolours = [0,0,0;jet(ma)];
colormap(jetcolours);

startz = min(size(src3,3),size(img3,3));
if reverse
    order = startz:-1:1;
else
    order = 1:startz;
end
for z = order
    set(0,'CurrentFigure',1)
    imagesc(src3(:,:,z), [srcmi srcma]);
    axis image off;
    set(0,'CurrentFigure',2)
    imagesc(img3(:,:,z), [mi ma]);
    axis image off;
    if z == order(1)
        pause(5);
    end
    pause(0.25);
end

    