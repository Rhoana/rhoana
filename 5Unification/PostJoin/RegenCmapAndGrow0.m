labels_dir = 'C:\dev\datasets\conn\main_dataset\5K_cube\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\res_from_0ct15_PF\FS=1\stitched\labels\';
grow_dir = 'C:\dev\datasets\conn\main_dataset\5K_cube\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\res_from_0ct15_PF\FS=1\stitched\labels_grow\';
cmap_filename = 'C:\dev\datasets\conn\main_dataset\5K_cube\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\res_from_0ct15_PF\FS=1\cmap.mat';

fprintf(1, 'Loading segmentation.\n');
seg_files = [ dir(fullfile(labels_dir, '*.tif')); ...
    dir(fullfile(labels_dir, '*.png')) ];
seg_files = sort({seg_files.name});

cmap = [];

% Load label data and compress
for zi = 1:length(seg_files)
    img = imread(fullfile(labels_dir, seg_files{zi}));
    if(size(img, 3)) == 3
        %Map 8-bit color image to 32 bit
        labels = uint32(img(:,:,1));
        labels = labels + uint32(img(:,:,2)) * 2^8;
        labels = labels + uint32(img(:,:,3)) * 2^16;

    else
        labels = uint32(img);
    end
    
    %Find and track unique labels
    ulabels = unique(labels(:));
    cmap = unique([cmap; ulabels]);
    fprintf(1, 'File %d. %d unique labels so far.\n', zi, size(cmap, 1));
    
    %Grow regions until there are no more black lines
    borders = labels==0;
    dnum = 0;
    disk1 = strel('disk', 1, 4);
    fprintf(1, 'Growing regions: ');
    while any(borders(:))
        dvol = imdilate(labels, disk1);
        labels(borders) = dvol(borders); 
        borders = labels==0;
        
        dnum = dnum + 1;
        fprintf(1, '%d,', dnum);
    end
    fprintf(1, '.\n');

    %Save
    color_labels = zeros(size(labels, 1), size(labels, 2), 3, 'uint8');
    color_labels(:,:,1) = uint8(bitand(labels, uint32(2^8-1)));
    color_labels(:,:,2) = uint8(bitand(bitshift(labels, -8), uint32(2^8-1)));
    color_labels(:,:,3) = uint8(bitand(bitshift(labels, -16), uint32(2^8-1)));
    imwrite(color_labels, fullfile(grow_dir, seg_files{zi}));
    
end

% choose a random ordering for colours
[~, new_order] = sort( rand(1, size(cmap, 1) - 1) );

cmap_labels = [cmap(1); cmap(new_order+1)];

cmap = zeros(length(cmap_labels), 3, 'uint8');
cmap(:,1) = uint8(bitand(cmap_labels, uint32(2^8-1)));
cmap(:,2) = uint8(bitand(bitshift(cmap_labels, -8), uint32(2^8-1)));
cmap(:,3) = uint8(bitand(bitshift(cmap_labels, -16), uint32(2^8-1)));

save(cmap_filename, 'cmap');

disp('Finished cmap regen.');

