p_to_dicing_lib = fullfile(pths.p_to_code, 'libs', 'utils', 'dicing');
addpath(p_to_dicing_lib);

# args in_image in_tree out_segmentations_in_hdf5 (with probabilities)

struct2var(cube_dicing_args);

if dicing_downsampling ~= 1,
    flg_downsampling = true;
else
    flg_downsampling = false;
end

%% If working with normal images
% Nobody parses diced_string, BUT, we DO parse
p_to_video_dataset                  = fullfile(pths.p_to_video_data, video_dataset_to_process);
p_to_video_folder                   = fullfile(p_to_video_dataset, video_to_process);
p_to_full_volume_input_images       = fullfile(p_to_video_folder, 'input_images');
p_to_full_volume_presegmentations   = fullfile(p_to_video_folder, 'pre_segs_kaynig');
p_to_diced                          = fullfile(p_to_video_folder, [diced_string]);
p_to_cubes                          = fullfile(p_to_diced, 'cubes');
p_to_diced_intradata_folder         = fullfile(p_to_diced, 'intra_cubes_data');
p_to_diced_interdata_folder         = fullfile(p_to_diced, 'inter_cubes_data');

makemydir(p_to_diced);
makemydir(p_to_cubes);
makemydir(p_to_diced_intradata_folder);
makemydir(p_to_diced_interdata_folder);

p_to_file = fullfile(p_to_diced, 'dicing_metadata.mat');
load(p_to_file, 'dicing_pars', 'small_dicing_pars', 'cube_names', 'locs', 'small_locs');

%% Getting the filenames
struct2var(locs);

lst_or_img_names      = dir(fullfile(p_to_full_volume_input_images, ['*' originals_read_img_extension]));
lst_or_img_names      = {lst_or_img_names(:).name};
% Remove the extension; this gives us more flexibility:
lst_or_img_names      = cellfun( @(x) remove_path_and_extension(x), ...
    lst_or_img_names, 'UniformOutput', false);

p_i                         = fullfile(p_to_full_volume_input_images, [lst_or_img_names{1} originals_read_img_extension]);
i_info                      = imfinfo(p_i);

full_mtdta.width  = i_info.Width;
full_mtdta.height = i_info.Height;
full_mtdta.nImgs  = numel(lst_or_img_names);

% z_indices correspond to frames:
disp('Reading filenames of the pre-segmentations');
list_of_tessellations = cell(nCubesZ,1);

for z_full_abs_ix = 1:full_mtdta.nImgs
    p_to_z_folder = fullfile(p_to_full_volume_presegmentations, lst_or_img_names{z_full_abs_ix});
    
    list_of_images_temp     = dir(fullfile(p_to_z_folder, ['*' kaynig_pre_segs_read_extension]));
    list_of_images_temp     = {list_of_images_temp(:).name};
    list_of_images_temp     = cellfun( @(x) remove_path_and_extension(x), ...
        list_of_images_temp, 'UniformOutput', false);
    list_of_tessellations{z_full_abs_ix}   = list_of_images_temp;
    list_of_tessellations{z_full_abs_ix}   = list_of_images_temp;
end

% Also do the image segmentation here

%Load the forest settings
%This file should be in libs/vkaynig (on the matlab path)
load('forest_TS1_TS3.mat','forest');

%Segmentation settings
threshRange = [0.26:0.01:0.5];
l_s_range = 0.6;
l_gc_range = 0.1;

maxSegi = length(threshRange) * length(l_s_range) * length(l_gc_range);

for z_full_abs_ix = 1:full_mtdta.nImgs
    
    name                      = [lst_or_img_names{z_full_abs_ix} originals_read_img_extension];
    p_i                       = fullfile(p_to_full_volume_input_images, name);
    i_or_full_z               = imread(p_i);
    
    %Do segmentation on full images
    %(or as large as possible with a very large overlap)
    imProb = generateMembraneProbabilities(i_or_full_z, forest);
    segs = gapCompletion(i_or_full_z, imProb, threshRange, l_s_range, l_gc_range);
    
