function compute_probabilities(varargin)

% Usage:
% compute_probabilities image_file_path forest_file_path out_file_path

% Computes boundary probabilities for  the input image using the
%   given random forest and predefined features.
% Outputs the resulting probabilities to out_file_path as an hdf5 file

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

%Enhance contrast (globally)
input_image = imadjust(input_image);

%Load the forest settings
load(forest_file_path,'forest');

% Generate features and calculate membrane probabilities
imProb = generateMembraneProbabilities(input_image, forest);

% avoid writing partial files
temp_file_path = [out_file_path, '_partial'];

% Save segmentations and probabilities
if exist(temp_file_path, 'file'),
  delete(temp_file_path);
end
h5create(temp_file_path, '/improb', [Inf, Inf], 'DataType', 'double', 'ChunkSize', [64,64], 'Deflate', 9, 'Shuffle', true);

h5write(temp_file_path, '/improb', imProb, [1, 1], size(imProb));

movefile(temp_file_path, out_file_path);
fprintf(1, 'compute_probabilities successfuly wrote to file: %s.\n', out_file_path);

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

