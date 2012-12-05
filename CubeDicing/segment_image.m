function segment_image(image_file_path, probs_file_path, out_file_path, xlo, ylo, xhi, yhi, core_xlo, core_ylo, core_xhi, core_yhi)
% Usage:
% segment_image image_file_in probability_file_in out_file_path

disp('Entry');

% Segments the input boundary probability image stored in an hdf5 file.
% Outputs the resulting segmentations to out_file_path as an hdf5 file

%Add the Segmentation lib folder to the path
d = fileparts(which(mfilename));
addpath(genpath(fullfile(d, '..', 'lib', 'segmentation')));

if ~exist(image_file_path, 'file')
    file_error(image_file_path);
end

if ~exist(probs_file_path, 'file')
    file_error(forest_file_path);
end

disp('segment_image starting\n');

%Open the input image
input_image = imread(image_file_path);

% Generate features and calculate membrane probabilities
imProb = h5read(probs_file_path, '/improb');

% Crop subregions
fprintf(1, 'Coring %d,%d to %d,%d', xlo+1, ylo+1, xhi, yhi);
input_image = input_image(xlo+1:xhi, ylo+1:yhi);
imProb = imProb(xlo+1:xhi, ylo+1:yhi);

% Segmentation settings (there are more settings in
% generateMembraneProbabilities and gapCompletion)

%Original settings
%threshRange = 0.2:0.01:0.5;
%l_s_range = 0.6;
%l_gc_range = 0.1;

%Dual settings: use 30 threshold and 30 gap completion segmentations
%(a linear combination of both might be even better)
%Threshold
threshRangeA = 0.21:0.01:0.5;
l_s_rangeA = 0.6;
l_gc_rangeA = 0.1;

%Gap completion
threshRangeB = 0.5;
l_s_rangeB = 0.2;
l_gc_rangeB = 0.05:0.05:1.5;

%maxSegi = length(threshRange) * length(l_s_range) * length(l_gc_range);
maxSegi = length(threshRangeA) * length(l_s_rangeA) * length(l_gc_rangeA) + ...
    length(threshRangeB) * length(l_s_rangeB) * length(l_gc_rangeB);


% Run segmentation

%Single mode
%segs = gapCompletion(chunk, chunk_prob, threshRange, l_s_range, l_gc_range);

%Dual mode
segs = cat(3, ...
           gapCompletion(input_image, imProb, threshRangeA, l_s_rangeA, l_gc_rangeA), ...
           gapCompletion(input_image, imProb, threshRangeB, l_s_rangeB, l_gc_rangeB));


% Extract the core
segs = segs(core_xlo+1-xlo:core_xhi-xlo, core_ylo+1-ylo:core_yhi-ylo, :);

%Convert
segs = uint8(segs) * 255;

% avoid writing partial files
temp_file_path = [out_file_path, '_partial'];

% Save segmentations and probabilities
if exist(temp_file_path, 'file'),
  delete(temp_file_path);
end
h5create(temp_file_path, '/segs', [Inf, Inf, Inf], 'DataType', 'uint8', 'ChunkSize', [64,64,10], 'Deflate', 9);
h5create(temp_file_path, '/original_coords', 6, 'DataType', 'uint32');

h5write(temp_file_path, '/segs', segs, [1, 1, 1], size(segs))
h5write(temp_file_path, '/original_coords', [core_xlo, core_ylo, 0, core_xhi, core_yhi, size(segs, 3)]);

movefile(temp_file_path, out_file_path);
fprintf(1, 'segment_image successfuly wrote to file: %s.\n', out_file_path);

return;

end


%Helper functions

function file_error(filename)
disp('Usage: segment_image image_file_path forest_file_path out_file_path');
disp(['Error: Input file does not exist: ' filename]);
error('File not found error.');
end
