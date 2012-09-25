function quickshowsclines (img3)
%Display a 3D image as a movie of 2D images
colormap([0,0,0;lines;lines;lines]);
mi = min(img3(:));
ma = max(img3(:));
if mi == ma
    ma = mi + 1;
end

for z = 1:size(img3,3)
    imagesc(img3(:,:,z), [mi ma]);
    pause(0.1);
end
    