labels_dir = 'C:\dev\datasets\conn\main_dataset\5K_cube\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\res_from_0ct15_PF\FS=1\stitched\labels\';
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
        labels = reshape(img, [size(img, 1) * size(img,2), 3]);
        ulabels = unique(labels, 'rows');
        cmap = unique([cmap; ulabels], 'rows');
    else
        error([ fullfile(seg_folder, seg_files{zi}), ' is not a color label image.' ]);
    end
    fprintf(1, 'File %d. Got %d labels.\n', zi, size(cmap, 1));
end

% choose a random ordering
[~, new_order] = sort( rand(1, size(cmap, 1) - 1) );

cmap = [cmap(1,:); cmap(new_order+1,:)];

save(cmap_filename, 'cmap');

disp('Finished cmap regen.');

