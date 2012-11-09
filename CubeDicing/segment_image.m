function segment_image(input_path, out_file_path)

% Usage:
% segment_image probability_file_in out_file_path

% Segments the input boundary probability image stored in an hdf5 file.
% Outputs the resulting segmentations to out_file_path as an hdf5 file

%Add the Segmentation lib folder to the path
d = fileparts(which(mfilename));
addpath(genpath(fullfile(d, '..', 'lib', 'segmentation')));


fprintf(1, 'segment_image starting\n');

% Generate features and calculate membrane probabilities
imProb = h5read(input_path, '/improb');

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

% Do gap completion and generate segmentations
%Single mode
%segs = gapCompletion(input_image, imProb, threshRange, l_s_range, l_gc_range);
%Dual mode
segs = cat(3, ...
    gapCompletion(input_image, imProb, threshRangeA, l_s_rangeA, l_gc_rangeA), ...
    gapCompletion(input_image, imProb, threshRangeB, l_s_rangeB, l_gc_rangeB));


%Convert
segs = uint8(segs) * 255;

% avoid writing partial files
temp_file_path = [out_file_path, '_partial'];

% Save segmentations and probabilities
if exist(temp_file_path, 'file'),
  delete(temp_file_path);
end
h5create(temp_file_path, '/segs', [Inf, Inf, Inf], 'DataType', 'uint8', 'ChunkSize', [64,64,10], 'Deflate', 9);

h5write(temp_file_path, '/segs', segs, [1, 1, 1], size(segs));

movefile(temp_file_path, out_file_path);
fprintf(1, 'segment_image successfuly wrote to file: %s.\n', out_file_path);

return;

end


%Helper functions

function arg_error
disp('Usage: segment_image image_file_path forest_file_path out_file_path');
error('Input argument error.');
end
