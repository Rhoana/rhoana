function makeBlocks (varargin)

sourceDirectory = 'C:\dev\datasets\Thousands_affine_fullres_fromDaniel\';
sourcePrefix = 'Thousand_highmag_affinefull_s';
sourceSuffix = '.png';
sourceDigits = 4;
sourceNumFormatString = sprintf('%%0%dd', sourceDigits);

targetDirectory = '';
targetDigits = 3;
targetNumFormatString = sprintf('%%0%dd', targetDigits);

startindex = 1;
endindex = 1850;

startpoint = [10000 10000 0]; % 0-indexed start point in the volume
blocksize = [512 512 10];
halo = [64 64 2];
nblocks = [2, 1, 2];

%Get input args
current_arg = 1;
nargs = length(varargin);
while current_arg <= nargs
    argstr = varargin{current_arg};
    switch lower(argstr)
        case {'-s', '-start'}
            if nargs > current_arg + 2
                current_arg = current_arg + 1;
                startpoint(1) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                startpoint(2) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                startpoint(3) = sscanf(varargin{current_arg}, '%d');
            else
                arg_error();
            end
        case {'-b', '-block'}
            if nargs > current_arg + 2
                current_arg = current_arg + 1;
                blocksize(1) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                blocksize(2) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                blocksize(3) = sscanf(varargin{current_arg}, '%d');
            else
                arg_error();
            end
        case {'-h', '-halo'}
            if nargs > current_arg + 2
                current_arg = current_arg + 1;
                halo(1) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                halo(2) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                halo(3) = sscanf(varargin{current_arg}, '%d');
            else
                arg_error();
            end
        case {'-n', '-nblocks'}
            if nargs > current_arg + 2
                current_arg = current_arg + 1;
                nblocks(1) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                nblocks(2) = sscanf(varargin{current_arg}, '%d');
                current_arg = current_arg + 1;
                nblocks(3) = sscanf(varargin{current_arg}, '%d');
            else
                arg_error();
            end
        case {'-i', '-indir'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                sourceDirectory = varargin{current_arg};
            else
                arg_error();
            end
        case {'-ip', '-prefix'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                sourcePrefix = varargin{current_arg};
            else
                arg_error();
            end
        case {'-is', '-suffix'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                sourceSuffix = varargin{current_arg};
            else
                arg_error();
            end
        case {'-o', '-outdir'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                targetDirectory = varargin{current_arg};
            else
                arg_error();
            end
    end
end

disp('Usage: makeBlocks [-start x y z] [-block x y z] [-halo x y z] [-nblocks x y z]');
disp('                  -i input_directory -prefix input_prefix -suffix input_suffix');
disp('                  -o output_directory');


targetDirectory = sprintf('%sstartpoint_%d,%d,%d_blocksize_%d,%d,%d_halo_%d,%d,%d_nblocks_%d,%d,%d\\', ...
    targetDirectory, startpoint, blocksize, halo, nblocks);

if (~exist(targetDirectory, 'dir'))
    mkdir(targetDirectory);
end

%Loop for each source image
startz = startindex + startpoint(3) - halo(3);
endz = startz + nblocks(3)*blocksize(3) - (nblocks(3)-1)*2*halo(3);

for imagez = startz:endz
    
    if imagez >= startindex && imagez <= endindex
        %Load the image
        sourceNum = sprintf(sourceNumFormatString, imagez);
        imgFilename = [sourceDirectory, sourcePrefix, sourceNum, sourceSuffix];
        fprintf(1, 'Opening file %s...\n', imgFilename);

        img = imread(imgFilename);
        
        %Normalize
        img = imadjust(img);
        
    else
        %Blank image
        img = zeros(blocksize(2), blocksize(1), 'uint8');
    end
    
    %Save sections to the correct directories
    for blockx = 1:nblocks(1)
        xi = (1:blocksize(1)) + startpoint(1) + (blockx-1)*blocksize(1) - halo(1) - (blockx-1)*2*halo(1);
        for blocky = 1:nblocks(2)
            yi = (1:blocksize(2)) + startpoint(2) + (blocky-1)*blocksize(2) - halo(2) - (blocky-1)*2*halo(2);
            
            if imagez >= startindex && imagez <= endindex
                saveImg = img(yi, xi);
            else
                %Blank image
                saveImg = img;
            end
            
            xdirstr = sprintf(targetNumFormatString, blockx-1);
            ydirstr = sprintf(targetNumFormatString, blocky-1);

            %Save to all appropriate blocks (repeats due to z halo)
            for blockz = 1:nblocks(3)
                zi = [1 blocksize(3)] + startpoint(3) + (blockz-1)*blocksize(3) - halo(3) - (blockz-1)*2*halo(3);
                if imagez >= zi(1) && imagez <= zi(2)
                    zdirstr = sprintf(targetNumFormatString, blockz-1);
                    blockdir = sprintf('%sblock_z%s_y%s_x%s\\', targetDirectory, zdirstr, ydirstr, xdirstr);
                    
                    imgdir = [blockdir 'images\'];
                    if ~exist(imgdir, 'dir')
                        mkdir(imgdir);
                    end
                    
                    featdir = [blockdir 'features\'];
                    if ~exist(featdir, 'dir')
                        mkdir(featdir);
                    end
                    
                    segdir = [blockdir 'segmentations\'];
                    if ~exist(segdir, 'dir')
                        mkdir(segdir);
                    end
                    
                    coldir = [blockdir 'colors\'];
                    if ~exist(coldir, 'dir')
                        mkdir(coldir);
                    end
                    
                    outputFilename = sprintf('%s%s.png', imgdir, sprintf(targetNumFormatString, imagez - zi(1)));
                    imwrite(saveImg, outputFilename);
                    
                    fprintf(1, 'Wrote file %s.\n', outputFilename);
                    
                end
            end
            
        end
    end
    
    clear img;
    
end
end

function arg_error
disp('Usage: makeBlocks [-start x y z] [-block x y z] [-halo x y z] [-nblocks x y z]');
disp('                  -d input_directory -prefix input_prefix -suffix input_suffix');
disp('                  -o output_directory');
error('Input argument error.');
end