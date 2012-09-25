function quickshowgrid3d (samples)

%Display a 3D image matrix as a movie of 2D images
colormap(gray);
ma = max(samples(:));
mi = min(samples(:));

a = size(samples,1);
d = a;

x = size(samples,3);
y = size(samples,4);

displaygrid = ones(x*d+x+1, y*d+y+1) .* mi;

%Show images in groups of x*y
for j = 0:y-1
    for i = 0:x-1
        start_x = i*(d+1)+2;
        start_y = j*(d+1)+2;

        %quickshowsc(samples(:,:,:,nimg));pause;
        displaygrid(start_x:start_x+d-1, start_y:start_y+d-1) = ...
            samples(:,:,i+1,j+1);
    end
end

imagesc(displaygrid, [mi ma]);
axis image
set(gca, 'xtick', ((d+1)/2:d+1:d*x)+1);
set(gca, 'ytick', ((d+1)/2:d+1:d*x)+1);

end