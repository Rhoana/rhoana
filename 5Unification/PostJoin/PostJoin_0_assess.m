addpath(fullfile('..', '..', '7Evaluation'));

dataset_dir = 'C:\dev\datasets\conn\main_dataset\';
%subvol_name = 'ac3train';
subvol_name = 'ac3test';
src_dir = fullfile(dataset_dir, subvol_name);
%src_dir = 'C:\dev\datasets\conn\main_dataset\ac3train\';

dice_string = 'diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1';
%result_name = 'res_from_sept_14_seg60_scf09_PF';
%result_name = 'res_from_sept_14_seg60_scf095_PF';
%result_name = 'res_from_sept_14_seg60_scf0975_PF';
%result_name = 'res_from_sept_30_minotrC_PF';
result_name = 'res_from_0ct02_PF';

nresult = 3;

%if ~exist('seg_vol', 'var')
%    fprintf(1, 'Loading working from PostJoin_Size.\n');
%    load(sprintf('PostJoin_Size_working_%s.mat', result_name));
%end

%ref_folder = 'C:\dev\datasets\groundtruth\gt_with_branching_regions_grown\train';
ref_folder = 'C:\dev\datasets\groundtruth\gt_with_branching_regions_grown\test';

% Prep for reference data

ref_files = [ dir(fullfile(ref_folder, '*.tif')); ...
    dir(fullfile(ref_folder, '*.png')) ];

ref_files = sort({ref_files.name});

ref_info = imfinfo(fullfile(ref_folder, ref_files{1}));

ref_vol = zeros(ref_info.Height, ref_info.Width, length(ref_files), 'uint32');

fprintf(1, 'Loading reference data (%d,%d,%d).\n', ...
    ref_info.Height, ref_info.Width, length(ref_files));

% Load reference data
for zi = 1:length(ref_files)
    img = imread(fullfile(ref_folder, ref_files{zi}));
    if(size(img, 3)) == 3
        %Map 8-bit color image to 32 bit
        ref_vol(:,:,zi) = img(:,:,1) + img(:,:,2) * 2^8 + img(:,:,3) * 2^16;
    else
        ref_vol(:,:,zi) = img;
    end
end


for minsegsize = [0 5000 10000 15000 20000 25000]
%for minsegsize = [1000 2500 5000 10000 15000 20000 25000]
    load(sprintf('PostJoin_%s_%s_FS%d_Size=%d.mat', subvol_name, result_name, nresult, minsegsize));
    %load(sprintf('PostJoin_%s_Size=%d.mat', result_name, minsegsize));
    score = andres_variation_of_information(segments, ref_vol);
    fprintf(1, 'Size=%d, Score=%1.4f.\n', minsegsize, score);
%     for maxjoinscore = [0.1 0.2 0.3 0.4 0.5]
%         load(sprintf('PostJoin_%s_Size=%d_Prob=%1.2f.mat', result_name, minsegsize, maxjoinscore));
%         score = andres_variation_of_information(segments, ref_vol);
%         fprintf(1, 'Size=%d, Prob=%1.2f, Score=%1.4f.\n', minsegsize, maxjoinscore, score);
%     end
end
