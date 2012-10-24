src_dir = 'C:\dev\datasets\conn\main_dataset\5k_cube\';
dice_string = 'diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1';
result_name = 'res_from_0ct02_PF';
%result_name = 'res_from_sept_14_seg60_scf095_PF';
%result_name = 'res_from_sept_14_seg60_scf0975_PF';
nresult = 1;

maxnfiles = 250;

%Load segmentation
fprintf(1, 'Loading segmentation.\n');
seg_folder = fullfile(src_dir, dice_string, result_name, ['FS=' num2str(nresult)], 'stitched', 'labels');
seg_files = [ dir(fullfile(seg_folder, '*.tif')); ...
    dir(fullfile(seg_folder, '*.png')) ];
seg_files = sort({seg_files.name});

if length(seg_files) > maxnfiles
    seg_files = seg_files(1:maxnfiles);
end

seg_info = imfinfo(fullfile(seg_folder, seg_files{1}));
seg_vol = zeros(seg_info.Height, seg_info.Width, length(seg_files), 'uint32');

% Load seg data
for zi = 1:length(seg_files)
    img = imread(fullfile(seg_folder, seg_files{zi}));
    if(size(img, 3)) == 3
        %Map 8-bit color image to 32 bit
        seg_vol(:,:,zi) = uint32(img(:,:,1));
        seg_vol(:,:,zi) = seg_vol(:,:,zi) + uint32(img(:,:,2)) * 2^8;
        seg_vol(:,:,zi) = seg_vol(:,:,zi) + uint32(img(:,:,3)) * 2^16;
    else
        seg_vol(:,:,zi) = img;
    end
    fprintf(1, '%d of %d.\n', zi, length(seg_files));
end

%Compress labels
[~, ~, seg_index] = unique(seg_vol(:));
seg_vol = reshape(uint32(seg_index)-1, size(seg_vol));

%Grow regions until there are no more black lines
borders = seg_vol==0;
dnum = 0;
disk1 = strel('disk', 1, 4);
while any(borders(:))
    dvol = imdilate(seg_vol, disk1);
    seg_vol(borders) = dvol(borders);
    borders = seg_vol==0;

    dnum = dnum + 1;
    fprintf(1, 'Growing regions: %d.\n', dnum);
end
