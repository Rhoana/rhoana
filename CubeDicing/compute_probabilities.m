function compute_probabilities(varargin)

% Usage:
% compute_probabilities image_file_path forest_file_path out_file_path [max_chunk_size]

% Computes boundary probabilities for  the input image using the
%   given random forest and predefined features.
% Outputs the resulting probabilities to out_file_path as an hdf5 file

%Add the Segmentation lib folder to the path
d = fileparts(which(mfilename));
addpath(genpath(fullfile(d, '..', 'lib', 'segmentation')));

% Check for errors
if length(varargin) < 3 || length(varargin) > 4
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

%Default chunk size
max_chunk_size = 512;
halo = 32;
if length(varargin) >= 4
    max_chunk_size = str2double(varargin{4});
end

if max_chunk_size < halo * 2
    fprintf(1, 'Error: max_chunk_size must be at least %d.\n', halo * 2);
    arg_error();
end

%Open the input image
input_image = imread(image_file_path);

%Greyscale only
if size(input_image, 3) > 1
    input_image = input_image(:,:,1);
end

%Enhance contrast (globally)
input_image = imadjust(input_image);

%Load the forest settings
load(forest_file_path,'forest');

%Allocate space for the probabilities
imsize = size(input_image);
imProb = zeros(imsize);

%Determine how many blocks to split this image into in X and Y
xdiv = 1;
ydiv = 1;
while floor(imsize(1) / xdiv) > max_chunk_size
    xdiv = xdiv + 1;
end
while floor(imsize(2) / ydiv) > max_chunk_size
    ydiv = ydiv + 1;
end

fprintf(1, 'Calculating probabilities in %dx%d chunks with %d overlap.\n', xdiv, ydiv, halo);

for xi = 1:xdiv
    for yi = 1:ydiv
        minX = floor(imsize(1) / xdiv * (xi-1)) + 1;
        maxX = floor(imsize(1) / xdiv * xi);
        minY = floor(imsize(2) / ydiv * (yi-1)) + 1;
        maxY = floor(imsize(2) / ydiv * yi);
        
        fprintf(1, 'Calculating region %d-%dx%d-%d.\n', minX, maxX, minY, maxY);
        
        if minX - halo <= 0
            haloMinX = 1;
            fromMinX = minX;
        else
            haloMinX = minX - halo;
            fromMinX = halo + 1;
        end
        
        if maxX + halo > imsize(1)
            haloMaxX = imsize(1);
            fromMaxX = maxX - haloMinX + 1;
        else
            haloMaxX = maxX + halo;
            fromMaxX = maxX - haloMinX + 1;
        end
        
        if minY - halo <= 0
            haloMinY = 1;
            fromMinY = minY;
        else
            haloMinY = minY - halo;
            fromMinY = halo + 1;
        end
        
        if maxY + halo > imsize(2)
            haloMaxY = imsize(2);
            fromMaxY = maxY - haloMinY + 1;
        else
            haloMaxY = maxY + halo;
            fromMaxY = maxY - haloMinY + 1;
        end
        
        fprintf(1, 'Halo region %d-%dx%d-%d.\n', haloMinX, haloMaxX, haloMinY, haloMaxY);
        
        chunk = input_image(haloMinX:haloMaxX, haloMinY:haloMaxY);
        
        %Calculate probabilities for this chunk
        chunk_prob = generateMembraneProbabilities(chunk, forest);

        %Assign (without halo) to the correct region of imProb
        imProb(minX:maxX, minY:maxY) = chunk_prob(fromMinX:fromMaxX, fromMinY:fromMaxY);
                
    end
end

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
disp('Usage: segment_image image_file_path forest_file_path out_file_path [max_chunk_size]');
error('Input argument error.');
end

function file_error(filename)
disp('Usage: segment_image image_file_path forest_file_path out_file_path [max_chunk_size]');
disp(['Error: Input file does not exist: ' filename]);
error('File not found error.');
end

