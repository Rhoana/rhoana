%     Copyright 2011 Seymour Knowles-Barley.
%
%     This file is part of Density2d.
% 
%     Density2d is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     Density2d is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with Density2d.  If not, see <http://www.gnu.org/licenses/>.


function density2d (varargin)

%defaults
filename = '';
directory = '';
outPrefix = '';
outSuffix = '_density2d';
close_edges = 1;
align = 0;
verbose = 0;
display = 0;
thresh = [];
randthresh = [];
step = [];
trials = [];
ds = [];

%threshold settings
threshmid = [10.5854 10.0236 10.0121 10.8900 9.8900];
threshstep = [0.0813 0.0557 0.0798 0 0];

%Rand optimized threshold settings
threshmidrand = [10.0550 9.9896 10.0998 10.0797 10.0506];

settings = struct( ...
    ... %general options
    'verbose', {false}, ...
    'display', {false}, ...
    ... %tracer options
    'randthresh', {false}, ...
    'thresh', {5}, ...
    'stepsize', {10}, ...
    'trials', {128}, ...
    'ds', {1}, ...
    'close_edges', {true}, ...
    'align', {false} ...
    );

%Read in command line arguments
current_arg = 1;
nargs = length(varargin);
while current_arg <= nargs
    argstr = varargin{current_arg};
    switch lower(argstr)
        case {'-f', '-file'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                filename = varargin{current_arg};
            else
                arg_error();
            end
        case {'-d', '-dir', '-directory'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                directory = varargin{current_arg};
            else
                arg_error();
            end
        case {'-op', '-outp', '-outprefix'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                outPrefix = varargin{current_arg};
            else
                arg_error();
            end
        case {'-os', '-outs', '-outsuffix'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                outSuffix = varargin{current_arg};
            else
                arg_error();
            end
        case {'-open', '-openedges'}
            close_edges = 0;
        case {'-a', '-align'}
            align = 1;
        case {'-t', '-thresh'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                if strcmpi('r', varargin{current_arg})
                    randthresh = true;
                else
                    thresh = sscanf(varargin{current_arg}, '%f');
                end
            else
                arg_error();
            end
        case {'-step'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                stepsize = sscanf(varargin{current_arg}, '%d');
            else
                arg_error();
            end
        case {'-rot', '-rotate'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                trials = sscanf(varargin{current_arg}, '%d');
            else
                arg_error();
            end
        case {'-scale'}
            if nargs > current_arg
                current_arg = current_arg + 1;
                ds = sscanf(varargin{current_arg}, '%f');
            else
                arg_error();
            end
        case {'-disp', '-display'}
            display = 1;
        case {'-v', '-verbose'}
            verbose = 1;
        otherwise
            arg_error();
    end
    current_arg = current_arg + 1;
end

%Sanity check
if isempty(filename) && isempty(directory)
    disp('Please specify an image file or directory.');
    arg_error();
end

%General settings
if close_edges == 0
    settings.close_edges = false;
end

if align == 1
    settings.align = true;
end

if display == 1
    settings.display = true;
end

if verbose
    settings.verbose = true;
end

if ~isempty(randthresh)
    settings.randthresh = randthresh;
end

if ~isempty(thresh)
    settings.thresh = thresh;
end

if ~isempty(step)
    settings.step = step;
end

if ~isempty(trials)
    settings.trials = trials;
end

if ~isempty(ds)
    settings.ds = ds;
end

%Open file and pre-process
if ~isempty(directory)
    %Find image list in this directory
    if directory(end) ~= filesep
        directory = [directory filesep];
    end
    %Find any png, jpg, jpeg, tif, tiff, bmp, gif
    files = dir([directory '*.png']);
    files = [files; dir([directory '*.jpg'])];
    files = [files; dir([directory '*.jpeg'])];
    files = [files; dir([directory '*.tif'])];
    files = [files; dir([directory '*.tiff'])];
    files = [files; dir([directory '*.bmp'])];
    files = [files; dir([directory '*.gif'])];
    if isempty(files)
        disp('No input image file(s) found in this directory.');
        arg_error();
    end
else
    %Find this image file
    files = dir(filename);
    if length(files) ~= 1 || files(1).isdir
        disp('Input image file not found.');
        arg_error();
    end
end

zd = 1;
cumu_off = [0;0];
all_patches = cell(1,length(files));

if settings.display
    fig1 = figure(1);
    set(gcf,'MenuBar', 'none');
    set(gcf,'ToolBar', 'none');
    scrsz = get(0,'ScreenSize');
    set(gcf,'Position',[(scrsz(3)-min(scrsz(3),1024))/2+50, (scrsz(4)-min(scrsz(4),768))/2+100, min(scrsz(3),1024)-100, min(scrsz(4),768)-100]);
    set(gca,'Position',[0.01 0.01 0.98 0.98])
end

for fi = 1:length(files)
    
    %Do tracing
    if ~isempty(directory)
        if files(fi).isdir
            continue;
        end
        fprintf(1, 'Tracing file %d of %d.\n', fi, length(files));
        imagefilename = [directory files(fi).name];
    else
        fprintf(1, 'Tracing %s.\n', filename);
        imagefilename = filename;
    end
    
    img = imread(imagefilename);
    img = img(:,:,1);
    imgsize = size(img);
    
    if settings.randthresh
        %Use the x vals optimized by Rand index
        x = threshmidrand;
    else
        %Thresh between 0 and 20 should give sensible results
        x = threshmid + (settings.thresh-10) .* threshstep;
    end
    
    start_thresh = x(1)-8;
    stop_thresh = x(2)-10;
    junc_thresh = x(3)-9;
    junc_window = x(4)-10+pi/2;
    gamma = x(5)-9;
    
    nlat = 0;
    latmaxang = pi/4;
    lat_inhibfac = 0;
    lat_excitefac = 0;
    repeat_costfac = 1;
    angle_costfac = 1;

    
%     [lines,  segs, best_activ, best_ai] = compare_draw_lines_norm_thresh_fix_gpu2_train(...
%         img, zd, [], stepsize, trials, start_thresh, stop_thresh, ...
%         junc_thresh, junc_window, gamma, nlat, latmaxang, ...
%         lat_inhibfac, lat_excitefac, repeat_costfac, angle_costfac);
% 
%     if nargin > 7 && closed
%         [lines,  segs] = closelines3(img, best_activ, best_ai, lines, segs, zd, stepsize);
%     end

    [lines,  segs, best_activ, best_ai] = density2d_draw_lines(...
        img, zd, [], settings.stepsize, settings.trials, start_thresh, stop_thresh, ...
        junc_thresh, junc_window, gamma, nlat, latmaxang, ...
        lat_inhibfac, lat_excitefac, repeat_costfac, angle_costfac, ...
        settings.display, settings.verbose);
    
    %Close lines (optional)
    if settings.close_edges
        [lines,  segs] = closelines3(img, best_activ, best_ai, lines, segs, zd, settings.stepsize, ...
                settings.display, settings.verbose);
        
        all_tssegs = segs.seg_index.values;
        if ~isempty(all_tssegs)
            all_tssegs = [all_tssegs{:}];
            [segscore, segorder] = sort([all_tssegs(:).index]);
            all_tssegs = all_tssegs(segorder);
            max_dist = 4*settings.stepsize;
            test_img = make_lineimg_closed(all_tssegs, [], [], imgsize, max_dist, 20+settings.stepsize);
        else
            test_img = false(size(img));
        end
        
    else
        
        all_tssegs = segs.seg_index.values;
        if ~isempty(all_tssegs)
            all_tssegs = [all_tssegs{:}];
            [segscore, segorder] = sort([all_tssegs(:).index]);
            all_tssegs = all_tssegs(segorder);
            max_dist = settings.stepsize;
            test_img = make_lineimg_closed(all_tssegs, [], [], imgsize, max_dist, 20+settings.stepsize);
        else
            disp('Sorry - no lines found.');
            test_img = false(size(img));
        end
        
    end
    
    %Export png file
    if ~isempty(directory)
        basename = [directory outPrefix files(fi).name outSuffix];
    else
        basename = [filename outSuffix];
    end
    imwrite(test_img, [basename '.png']);
    
    %Align with previous trace
    if settings.align && zd > 1
        fprintf(1, 'Aligning with previous trace.\n');
        [offsets, cumu_off, patches] = dp_align_loop(lines, lines_prev, zd, cumu_off, settings);
        
        % Output the results (images are output with no offset)
        offsetfile = fopen([basename '.offsets.csv'], 'w');
        fprintf(offsetfile, 'step_x_offset,%1.2f\nstep_y_offset,%1.2f\n', offsets(1), offsets(2));
        fprintf(offsetfile, 'cumulative_x_offset,%1.2f\ncumulative_y_offset,%1.2f\n', cumu_off(1), cumu_off(2));
        fclose(offsetfile);
        
        %%%% 3D view is slow can can cause compatibility issues on some
        %%%% versions of Matlab - disabled. (edit below and dp_align_loop.m
        %%%% to enable)
%         if settings.display
%             all_patches{fi} = patches;
%             fig2 = figure(2);
%             set(gcf,'MenuBar', 'none');
%             set(gcf,'ToolBar', 'none');
%             %set(gcf,'Position',[30 50 scrsz(3)-90 scrsz(4)-90]);
%             set(gcf,'Position',[(scrsz(3)-min(scrsz(3),1024))/2+50, (scrsz(4)-min(scrsz(4),768))/2+100, min(scrsz(3),1024)-100, min(scrsz(4),768)-150]);
%             set(gca,'Position',[0.01 0.01 0.98 0.98])
%             
%             draw_patches(all_patches);
%                 
%             close(fig2);
%         end
    end
    
    lines_prev = lines;
    zd = zd + 1;
    
end

close(fig1);

disp('Tracing complete.');

end

function arg_error
%disp('Input argument error.');
disp('Usage: density2d (-f filename | -d directory) -op outputPrefix -os outputSuffix');
disp('                    [-display] [-verbose] [-thresh #.##] [-step #] [-rot #] [-scale #.##] [-open] [-align]');
error('Input argument error.');
end