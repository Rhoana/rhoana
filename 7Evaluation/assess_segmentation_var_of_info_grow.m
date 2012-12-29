%function score = assess_segmentation_var_of_info(seg_folder_list, ref_folder)
 
%seg_prefix = 'C:\dev\datasets\conn\main_dataset\ac3train\diced_xy=256_z=16_xyOv=128_zOv=8_dwnSmp=1';
%seg_prefix = 'C:\dev\datasets\conn\main_dataset\ac3train\diced_xy=384_z=36_xyOv=128_zOv=12_dwnSmp=1';
seg_prefix = 'C:\dev\datasets\conn\main_dataset\ac3train\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1';
%seg_prefix = 'C:\dev\datasets\conn\main_dataset\ac3test\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1';

% result_list = {'res_from_sept_3_tuatara_25seg_PF', ...
%     'res_from_sept_14_scf075_PF', ...
%     'res_from_sept_14_scf08_PF', ...
%     'res_from_sept_14_scf085_PF', ...
%     'res_from_sept_14_scf09_PF', ...
%     'res_from_sept_14_scf095_mop0_PF', ...
%     'res_from_sept_14_scf10_PF', ...
%     'res_from_sept_14_scf11_PF', ...
%     'res_from_sept_14_scf12_PF', ...
%     };

% result_list = {'res_from_sept_14_seg60_scf0975_PF', ...
%     'res_from_sept_14_scf095_PF', ...
%     'res_from_sept_20_segGC30_PF'};

%result_list = {...%'res_from_sept_14_seg60_scf0975_PF', ...
%     ...'res_from_sept_28_otr899_PF', ...
%     ...%'res_from_sept_28_nmb100_PF', ...
%     ...%'res_from_sept_28_nmb10_PF', ...
%     ...'res_from_sept_28_nmb1_PF', ...
%     ...'res_from_sept_30_minotr_PF', ...
%     ...'res_from_sept_30_minotrB_PF', ...
%     'res_from_sept_30_minotrC_PF', ...
%     ...%'res_from_ODY_sept_27_PF'
%     };

%Test (512)
%result_list = {'res_from_0ct02_PF'};

%Small dicing (256)
%result_list = {'res_from_SmallDice_PF'};
%result_list = {'res_from_SmDi_J2_PF'}; %More joining - works well on test (not so good on the full cube)
%result_list = {'res_from_SmDi_J1_scf085_PF'};
%result_list = {'res_from_SmDi_J1_scf08_PF'};

%Medium dicing (384)
%result_list = {'res_from_MedDi_scf09_PF'};

%Large dicing (512)
%result_list = {'res_from_sept_14_scf09_PF'};

%Good Branching 512
%result_list = {'res_from_ODY_sept_27_PF'};
%result_list = {'res_from_sept_30_minotrC_PF'}; % This is the best result (so far) for training data

result_list = {'res_from_Dec19_0925_60_PF'};
%result_list = {'res_from_Dec19_09_60_PF'};

% result_list = {'res_from_sept_14_scf095_mop0_PF', ...
%     'res_from_sept_14_scf095_mop1000_PF', ...
%     'res_from_sept_14_scf095_mop1000_joinall_PF'};

nresults = 4;

seg_folder_list = cell(1,nresults);

ref_folder = 'C:\dev\datasets\groundtruth\gt_with_branching_regions_grown\train';
%ref_folder = 'C:\dev\datasets\groundtruth\gt_with_branching_regions_grown\test';


% Assess the seg_folder 3D segmentation against the ref_folder reference
% Use variation of information to determine score

% Expects folders of .tif or .png images
% Each image should be a single slice, in sort order


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

scores = zeros(length(result_list), length(seg_folder_list));

for result_i = 1:length(result_list)
    
    for fs = 1:nresults
        seg_folder_list{fs} = fullfile(seg_prefix, result_list{result_i}, ['FS=' num2str(fs)], 'stitched', 'labels');
    end

    for seg_folder_i = 1:length(seg_folder_list)
        seg_folder = seg_folder_list{seg_folder_i};

        % Prep for seg data
        seg_files = [ dir(fullfile(seg_folder, '*.tif')); ...
            dir(fullfile(seg_folder, '*.png')) ];

        seg_files = sort({seg_files.name});

        seg_info = imfinfo(fullfile(seg_folder, seg_files{1}));


        %fprintf(1, 'Loading segmentation %d data (%d,%d,%d).\n', ...
        %    seg_folder_i, seg_info.Height, seg_info.Width, length(seg_files));


        if (ref_info.Height ~= seg_info.Height || ref_info.Width ~= seg_info.Width || ...
                length(ref_files) ~= length(seg_files))
            fprintf(1, 'WARNING: Segmentation %d. Inputs are different sizes - skipping.\n', seg_folder_i);
            scores(result_i, seg_folder_i) = NaN;
            continue;
        end


        % Assume all images are the same size
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
        end
        
        disk1 = strel('disk', 1, 4);
        
        %Grow the regions so that there are no seperating pixels
        borders = seg_vol==0;
        dnum = 0;
        while any(borders(:))
            dvol = imdilate(seg_vol, disk1);
            seg_vol(borders) = dvol(borders);
            borders = seg_vol==0;
            
            dnum = dnum + 1;
            %fprintf(1, 'Growing regions: %d.\n', dnum);
        end

    %     %Compress labels (required for rand index)
    %     [~, ~, seg_index] = unique(seg_vol(:));
    %     seg_vol = reshape(uint32(seg_index)-1, size(seg_vol));
    % 
    %     %Rand Index Calculation
    %     %Ignore boundaries
    %     ignore = (seg_vol==0) | (ref_vol == 0);
    %     scores(seg_folder_i) = RandIndex(seg_vol(~ignore), ref_vol(~ignore));

        %Variation of information calculation
        scores(result_i, seg_folder_i) = andres_variation_of_information(seg_vol, ref_vol);

        fprintf(1, 'Segmentation %d has a score of %1.4f.\n', seg_folder_i, scores(result_i, seg_folder_i));
        
    end
    
    fprintf(1, 'Mean segmentation score of %1.4f.\n\n', mean(scores(result_i, :)));
    
end


