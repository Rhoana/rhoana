src_dir = 'C:\dev\datasets\NewPipelineResults0\labels';
dest_dir = 'C:\dev\datasets\NewPipelineResults0\labels_grow';

maxnfiles = 52;

%Load segmentation
fprintf(1, 'Loading segmentation.\n');
seg_files = [ dir(fullfile(src_dir, '*.tif')); ...
    dir(fullfile(src_dir, '*.png')) ];
seg_files = sort({seg_files.name});

if length(seg_files) > maxnfiles
    seg_files = seg_files(1:maxnfiles);
end

seg_info = imfinfo(fullfile(src_dir, seg_files{1}));
seg_vol = zeros(seg_info.Height, seg_info.Width, 'uint32');

% Load seg data
for zi = 1:length(seg_files)
    img = imread(fullfile(src_dir, seg_files{zi})) - 1;
    if(size(img, 3)) == 3
        %Map 8-bit color image to 32 bit
        seg_vol(:,:) = uint32(img(:,:,1));
        seg_vol(:,:) = seg_vol(:,:,zi) + uint32(img(:,:,2)) * 2^8;
        seg_vol(:,:) = seg_vol(:,:,zi) + uint32(img(:,:,3)) * 2^16;
    else
        seg_vol(:,:) = img;
    end
    fprintf(1, '%d of %d.\n', zi, length(seg_files));
    
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

    fprintf(1, 'Saving...\n');
    imwrite( seg_vol, fullfile(dest_dir, seg_files{zi}) );
    
end
