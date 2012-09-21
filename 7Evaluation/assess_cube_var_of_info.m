%function score = assess_segmentation_var_of_info(seg_folder_list, ref_folder)

cube_folder = 'C:\dev\datasets\conn\main_dataset\ac3train\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\cubes';

%result_string = 'sept_3_tuatara_25seg';
%result_string = 'sept_14_scf075';
%result_string = 'sept_14_scf08';
%result_string = 'sept_14_scf085';
%result_string = 'sept_14_scf095';
result_string = 'sept_20_segGC30';

disp(result_string);

ref_folder = 'C:\dev\datasets\groundtruth\gt_with_branching_regions_grown\train';


% Assess the cube_folder 3D segmentations against the ref_folder reference
% Use variation of information to determine score

% Reads cube dicing folder parameters and splits reference accordingly


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


%Find cube folders
cubes_dir_list = dir(fullfile(cube_folder, 'cubeId=*'));

scores = zeros(1, length(cubes_dir_list));

for cubei = 1:length(cubes_dir_list)
    %Parse directory name for cube location
    cube_dir = cubes_dir_list(cubei).name;
    scan_result = sscanf(cube_dir, 'cubeId=%d_Z=%d_Y=%d_X=%d_minZ=%d_maxZ=%d_minY=%d_maxY=%d_minX=%d_maxX=%d_dwnSmp=%d');
    cubeid = scan_result(1);
    minZ = scan_result(5);
    maxZ = scan_result(6);
    minY = scan_result(7);
    maxY = scan_result(8);
    minX = scan_result(9);
    maxX = scan_result(10);
    
    %fprintf(1, 'Scoring sub-cube %d starting at (%d,%d,%d).\n', cubeid, minX, minY, minZ);
    
    seg_folder = fullfile(cube_folder, cube_dir, 'results', result_string, 'indep_cube_rendering', 'labels');
    
    seg_files = dir(fullfile(seg_folder, '*.png'));
    
    seg_files = sort({seg_files.name});

    seg_info = imfinfo(fullfile(seg_folder, seg_files{1}));


    %fprintf(1, 'Loading cube %d data (%d,%d,%d).\n', ...
    %    cubeid, seg_info.Height, seg_info.Width, length(seg_files));


    if ((maxX - minX + 1) ~= seg_info.Height || (maxY - minY + 1) ~= seg_info.Width || ...
            (maxZ - minZ + 1) ~= length(seg_files))
        fprintf(1, 'WARNING: Cube %d. Unexpected input size - skipping.\n', cubeid);
        scores(cubei) = NaN;
        continue;
    end


    % Assume all images are the same size
    seg_vol = zeros(seg_info.Height, seg_info.Width, length(seg_files), 'uint32');

    ref_subvol = ref_vol(minY:maxY, minX:maxX, minZ:maxZ);

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
    end
    
%     %Compress labels (required for rand index)
%     [~, ~, seg_index] = unique(seg_vol(:));
%     seg_vol = reshape(uint32(seg_index)-1, size(seg_vol));
% 
%     %Rand Index Calculation
%     %Ignore boundaries
%     ignore = (seg_vol==0) | (ref_subvol == 0);
%     scores(cubei) = RandIndex(seg_vol(~ignore), ref_subvol(~ignore));

    %Variation of information calculation
    scores(cubei) = andres_variation_of_information(seg_vol, ref_subvol);

    fprintf(1, 'Cube %d has a score of %1.4f.\n', cubeid, scores(cubei));
    
end


fprintf(1, 'Mean segmentation score of %1.4f.\n\n', mean(scores));


