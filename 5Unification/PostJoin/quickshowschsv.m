function quickshowschsv (img3)
%Display a 3D image as a movie of 2D images
nsegments = max(img3(:));
reps = ceil(nsegments/length(lines));
colormap([0,0,0;repmat(hsv,reps,1)]);
mi = min(img3(:));
ma = max(img3(:));
if mi == ma
    ma = mi + 1;
end

for z = 1:size(img3,3)
    imagesc(img3(:,:,z), [mi ma]);
    pause(0.1);
end
    