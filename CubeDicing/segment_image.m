function segment_image(varargin)

% Usage:
% segment_image image_file_path forest_file_path out_file_path

% Segments the input image using the given random forest and predefined features
% Outputs the resulting probabilities and segmentations to out_file_path as an hdf5 file

%Add the Segmentation lib folder to the path
d = fileparts(which(mfilename));
addpath(genpath(fullfile(d, '..', 'lib', 'segmentation')));

% Check for errors
if length(varargin) ~= 3
    arg_error();
end

fprintf(1, 'segment_image starting\n');

image_file_path = varargin{1};
forest_file_path = varargin{2};
out_file_path = varargin{3};

if ~exist(image_file_path, 'file')
    file_error(image_file_path);
end

if ~exist(forest_file_path, 'file')
    file_error(forest_file_path);
end


%Open the input image
input_image = imread(image_file_path);



%Load the forest settings
load(forest_file_path,'forest');


% Segmentation settings (there are more settings in
% generateMembraneProbabilities and gapCompletion)
threshRange = [0.26:0.01:0.5];
l_s_range = 0.6;
l_gc_range = 0.1;
%maxSegi = length(threshRange) * length(l_s_range) * length(l_gc_range);


% Generate features and calculate membrane probabilities
imProb = generateMembraneProbabilities(input_image, forest);

% Do gap completion and generate segmentations
segs = gapCompletion(input_image, imProb, threshRange, l_s_range, l_gc_range);

%Convert
segs = uint8(segs) * 255;

% avoid writing partial files
temp_file_path = [out_file_path, '_partial'];

% Save segmentations and probabilities
if exist(temp_file_path, 'file'),
  delete(temp_file_path);
end
h5create(temp_file_path, '/improb', [Inf, Inf], 'DataType', 'double', 'ChunkSize', [64,64], 'Deflate', 9, 'Shuffle', true);
h5create(temp_file_path, '/segs', [Inf, Inf, Inf], 'DataType', 'uint8', 'ChunkSize', [64,64,10], 'Deflate', 9);

h5write(temp_file_path, '/improb', imProb, [1, 1], size(imProb));
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

function file_error(filename)
disp('Usage: segment_image image_file_path forest_file_path out_file_path');
disp(['Error: Input file does not exist: ' filename]);
error('File not found error.');
end

