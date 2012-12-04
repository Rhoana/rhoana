function compute_probabilities(image_file_path, forest_file_path, out_file_path, xlo, ylo, xhi, yhi, core_xlo, core_ylo, core_xhi, core_yhi)

% Usage:
% compute_probabilities image_file_path forest_file_path out_file_path xlo, ylo, xhi, yhi, core_xlo, core_ylo, core_xhi, core_yhi

% Computes boundary probabilities for  the input image using the
%   given random forest and predefined features.
% Computes in the region [xlo+1:xhi, ylo+1:yhi] and stores the region [core_xlo+1:core_xhi, core_ylo+1:core_yhi].
% Outputs the resulting probabilities to out_file_path as an hdf5 file

%Add the Segmentation lib folder to the path
d = fileparts(which(mfilename));
addpath(genpath(fullfile(d, '..', 'lib', 'segmentation')));

fprintf(1, 'compute_probabilities starting\n');

if ~exist(image_file_path, 'file')
    file_error(image_file_path);
end

if ~exist(forest_file_path, 'file')
    file_error(forest_file_path);
end

%Open the input image
input_image = imread(image_file_path);

%Greyscale only
if size(input_image, 3) > 1
    input_image = input_image(:,:,1);
end

%Enhance contrast (globally)
input_image = imadjust(input_image);

% Crop subregion - Use python indexing
fprintf(1, 'Coring %d,%d to %d,%d', xlo+1, ylo+1, xhi, yhi);
input_image = input_image(xlo+1:xhi, ylo+1:yhi);

%Load the forest settings
load(forest_file_path,'forest');

%Compute probabilities
imProb = generateMembraneProbabilities(input_image, forest);

% Extract the core
imProb = imProb(core_xlo+1-xlo:core_xhi-xlo, core_ylo+1-ylo:core_yhi-ylo);

% avoid writing partial files
temp_file_path = [out_file_path, '_partial'];

% Save segmentations and probabilities
if exist(temp_file_path, 'file'),
  delete(temp_file_path);
end
h5create(temp_file_path, '/improb', [Inf, Inf], 'DataType', 'double', 'ChunkSize', [64,64], 'Deflate', 9, 'Shuffle', true);
h5create(temp_file_path, '/original_coords', 4, 'DataType', 'uint32');

h5write(temp_file_path, '/improb', imProb, [1, 1], size(imProb));
h5write(temp_file_path, '/original_coords', [core_xlo, core_ylo, core_xhi, core_yhi]);

movefile(temp_file_path, out_file_path);
fprintf(1, 'compute_probabilities successfuly wrote to file: %s.\n', out_file_path);

return;

end


%Helper functions

function arg_error
% 
disp('Usage: compute_probabilities image_file_path forest_file_path out_file_path xlo, ylo, xhi, yhi, core_xlo, core_ylo, core_xhi, core_yhi')
error('Input argument error.');
end

function file_error(filename)
disp('Usage: segment_image image_file_path forest_file_path out_file_path [max_chunk_size]');
disp('   or: segment_image image_file_path forest_file_path out_file_path [xlo ylo xhi yhi]');
disp(['Error: Input file does not exist: ' filename]);
error('File not found error.');
end

