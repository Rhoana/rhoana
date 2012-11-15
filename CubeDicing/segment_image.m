function segment_image(image_file_path, probs_file_path, out_file_path)
% Usage:
% segment_image image_file_in probability_file_in out_file_path

% Segments the input boundary probability image stored in an hdf5 file.
% Outputs the resulting segmentations to out_file_path as an hdf5 file

%Add the Segmentation lib folder to the path
d = fileparts(which(mfilename));
addpath(genpath(fullfile(d, '..', 'lib', 'segmentation')));

if ~exist(image_file_path, 'file')
    file_error(image_file_path);
end

disp('segment_image starting\n');

%Open the input image
input_image = imread(image_file_path);

% Generate features and calculate membrane probabilities
imProb = h5read(probs_file_path, '/improb');

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

%maxchunksizeXY = 512;
%halo = 128;
maxchunksizeXY = 1024;
halo = 256;

%Allocate space for segmentations
imsize = size(input_image);
segs = zeros(imsize(1), imsize(2), maxSegi, 'uint8');

xdiv = 1;
ydiv = 1;

while floor(imsize(1) / xdiv) > maxchunksizeXY
    xdiv = xdiv + 1;
end

while floor(imsize(2) / ydiv) > maxchunksizeXY
    ydiv = ydiv + 1;
end

fprintf(1, 'Segmenting image in %dx%d chunks with %d overlap.\n', xdiv, ydiv, halo);


for xi = 1:xdiv
    for yi = 1:ydiv
        minX = floor(imsize(1) / xdiv * (xi-1)) + 1;
        maxX = floor(imsize(1) / xdiv * xi);
        minY = floor(imsize(2) / ydiv * (yi-1)) + 1;
        maxY = floor(imsize(2) / ydiv * yi);
        
        fprintf(1, 'Segmenting region %d-%dx%d-%d.\n', minX, maxX, minY, maxY);
        
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
        
        %Run Segmentation on this chunk
        chunk_prob = imProb(haloMinX:haloMaxX, haloMinY:haloMaxY);
                
        %Single mode
        %segs = gapCompletion(chunk, chunk_prob, threshRange, l_s_range, l_gc_range);
        
        %Dual mode
        chunk_segs = cat(3, ...
            gapCompletion(chunk, chunk_prob, threshRangeA, l_s_rangeA, l_gc_rangeA), ...
            gapCompletion(chunk, chunk_prob, threshRangeB, l_s_rangeB, l_gc_rangeB));
        
        %Assign to the correct region
        segs(minX:maxX, minY:maxY, :) = uint8(chunk_segs(fromMinX:fromMaxX, fromMinY:fromMaxY, :));
        
    end
end


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

function file_error(filename)
disp('Usage: segment_image image_file_path forest_file_path out_file_path');
disp(['Error: Input file does not exist: ' filename]);
error('File not found error.');
end
