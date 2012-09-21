function dice_block(varargin)

% Usage:
% dice_block xmin ymin xmax ymax segfile1.hdf5 segfile2.hdf5 ... out_file_path

% Extracts a cube from [xmin:xmax, ymin:ymax] and from each segmentation file (assumed to be stacked in Z).
% Outputs the resulting segmentations to outfile as a 4D array indexed by (X, Y, Z, segmentation)

% Check for errors
if length(varargin) < 6
    arg_error();
end

xmin = str2num(varargin{1});
ymin = str2num(varargin{2});
xmax = str2num(varargin{3});
ymax = str2num(varargin{4});
out_file_path = varargin{length(varargin)};
num_slices = length(varargin) - 5;

xsize = xmax - xmin + 1;
ysize = ymax - ymin + 1;

% Save out subset of segmentations
if ~exist(out_file_path, 'file'),
  h5create(out_file_path, '/cubesegs', [Inf, Inf, Inf, Inf], 'DataType', 'uint8', 'ChunkSize', [64, 64, 4, 10], 'Deflate', 9, 'Shuffle', true);
end

fprintf(1, ['dice_block starting. ', num2str(num_slices), ' slices.\n']);

for i=1:num_slices,
  dat = h5read(varargin{4 + i}, '/segs', [xmin, ymin, 1], [xsize, ysize, Inf]);
  [xs, ys, ss] = size(dat);
  dat = reshape(dat, [xs, ys, 1, ss]);
  h5write(out_file_path, '/cubesegs', dat, [1, 1, i, 1], [xs, ys, 1, ss]);
end

fprintf(1, 'dice_block successfuly wrote to file: %s.\n', out_file_path);

return;

end


%Helper functions

function arg_error
disp('Usage:  dice_block xmin ymin xmax ymax segfile1.hdf5 segfile2.hdf5 ... out_file_path');
error('Input argument error.');
end

function file_error(filename)
disp('Usage:  dice_block xmin ymin xmax ymax segfile1.hdf5 segfile2.hdf5 ... out_file_path');
disp(['Error: Input file does not exist: ' filename]);
error('File not found error.');
end

