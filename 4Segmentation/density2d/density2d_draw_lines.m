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


function [mem_lines, segment_index, best_activ, best_ai] = ...
    density_draw_lines( ...
    img, zi, segment_index, stepsize, trials, start_thresh, stop_thresh, ...
    start_junction_thresh, junction_window, gamma, nlat, latmaxang, ...
    lat_inhibfac, lat_excitefac, repeat_costfac, angle_costfac, display, verbose)

%draw lines to match the membrane

%close all;
display_progress = display;
display_working = 0;
display_junc_working = 0;
%verbose = false;

ds = 1;
r = 20;
w = 2*r+1;

wwwy = size(img,1);
wwwx = size(img,2);

%Size of location step (pixels)
%stepsize = 10;

%Max number of steps (in each direction)
nsteps = 1000;

%gamma
%gamma = 1;

% Start candidate threshold (standard deviations)
% stop_thresh = 0.4;
% start_thresh = 2;
%stop_thresh = 0.2;
%start_thresh = 0.5;
%stop_thresh = start_thresh/5;

%Test several orientations and pick the best
%trials = 12;
a_min = 0;
a_max = pi;
a_interval = (a_max-a_min) / trials;
%thicknesses = (5:20)*ds;

%Junction thresholds
%start_junction_thresh = 1.0;
%junction_window = pi/2;
min_junction_angle = (pi-junction_window)/2;
min_junction_a = ceil((min_junction_angle - a_min) / a_interval + 1);
junction_nmax = 1;
junction_radius = ceil(stepsize/2);
[xg yg] = meshgrid((-junction_radius:junction_radius));
junction_area = sqrt(xg.^2 + yg.^2);
junction_area = junction_area <= junction_radius;

startpoint_radius = stepsize*2;
[xg yg] = meshgrid((-startpoint_radius:startpoint_radius));
startpoint_area = sqrt(xg.^2 + yg.^2);
startpoint_area = startpoint_area <= startpoint_radius;

mitothresh = 200;
synthresh = 250;

%Angle cost function
%TODO Tune angle cost function (based on prior expectation?)
%angle_cost = (exp((0:1/trials:1-1/trials)*5)-1) / 90;

%Create or load the filters
%figure(1);
if display_progress
    set(0,'CurrentFigure',1)
    subplot(2,3,4);
    set(gca,'Position',[0.01 0.01 0.32 0.49])
    axis image off;
    title('Calculating...');
end
%[filters filt_fuzz filt_all class_syn class_mito] = generate_filters_nn_g(ds, r, trials, 1, gamma);
[filters filt_fuzz filt_all class_syn class_mito] = generate_filters_nn_g_r(ds, r, trials, 1, gamma);
%[filters filt_fuzz filt_all class_syn class_mito] = generate_filters_nn_g_p(ds, r, trials, 1, gamma);
if display_progress
    pause(0.1);
end

%init storage for results
if isempty(segment_index)
    segment_index = SegmentIndex();
end

%Scale between -1 and 1 (optional?)
%(Better for thickness 'detection')
%large_img = (double(img)-128)/128;

%Normalise to fill between -1 and 1
large_img = (double(img)-double(min(img(:))));
large_img = (large_img ./ max(large_img(:))) .* 2 - 1;

if display_progress
    subplot(2,3,1);
    colormap(gray);
    %imagesc(large_img);
    imagesc(img);
    axis image off;
    set(gca,'Position',[0.01 0.51 0.32 0.49])
    xlim([1+r, wwwx-r]);
    ylim([1+r, wwwy-r]);
    subplot(2,3,[2 3 5 6]);
    colormap(gray);
    imagesc(large_img);
    axis image off;
    set(gca,'Position',[0.34 0.01 0.65 0.99])
    xlim([1+r, wwwx-r]);
    ylim([1+r, wwwy-r]);
    pause(0.1);
end

%Find some good starting points

%Filter the image

activ = zeros(wwwy, wwwx, trials);
%activ_fuzz = zeros(www, www, trials);

%%%%%%%%%%%%%%%%%%%%
%GPU FFT Convolution
%%%%%%%%%%%%%%%%%%%%

% if ~libisloaded('MultiConvolveFFT2D_CUDA')
%     loadlibrary('C:\dev\convolution\MultiConvolveFFT2D_CUDA\x64\Release\MultiConvolveFFT2D_CUDA.dll', @MultiConvolveFFT2D_CUDA_proto, 'alias', 'MultiConvolveFFT2D_CUDA');
% end
% 
% %Do all images in one step
% 
% pres = libpointer('singlePtr', activ);
% pimg = libpointer('singlePtr', large_img);
% pfilt = libpointer('singlePtr', filters);
% 
% calllib('MultiConvolveFFT2D_CUDA', 'MultiConvolveFFT2D', pres, ...
%     pfilt, size(filters,1), size(filters,2), size(filters,3), ...
%     pimg, size(large_img,1), size(large_img,2), size(large_img,3));
% 
% result = get(pres, 'Value');
% 
% for a = 1:trials
%     activ(:,:,a) = result(:,(1:size(large_img,2)) + size(large_img,2)*(a-1));
% end

%%%%%%%%%%%%%%%%%%%%
%CPU FFT Convolution
%%%%%%%%%%%%%%%%%%%%

%Prepare for fft
fs = [0,0];
for dim = 1:2
    fs(dim) = size(large_img,dim)+size(filters,dim)-1;
    %fs(dim) = 2^nextpow2(size(a,dim)+size(b,dim)-1);
end
%Precalc image fft
imgfft = large_img;
for d = 1:2
    imgfft = fft(imgfft,fs(d),d);
end

for a = 1:trials
    %activ(:,:,a) = conv2(large_img, filters(:,:,a), 'same');
    %activ_fuzz(:,:,a) = conv2(large_img, filt_fuzz(:,:,a), 'same');
    %activ_all(:,:,a) = conv2(large_img, filt_all(:,:,a), 'same');
    
    %Do fft convolution
    bf = filters(:,:,a);
    for d = 1:2
        bf = fft(bf,fs(d),d);
    end
    fftresult = imgfft.*bf;
    for d = 1:2
        fftresult = ifft(fftresult,fs(d),d);
    end
    midb1 = floor(size(filters,1)/2);
    midb2 = floor(size(filters,2)/2);
    fromi1 = midb1+1:midb1+size(large_img,1);
    fromi2 = midb2+1:midb2+size(large_img,2);
    activ(:,:,a) = real(fftresult(fromi1,fromi2));
    
end

%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%

%Choose the best angles;
[best_activ best_ai]= max(activ,[],3);
%[best_activ_fuzz best_ai_fuzz]= max(activ_fuzz,[],3);

ba_mean = mean(best_activ(:));
ba_sd = std(best_activ(:));
ba_sd0 = ba_sd;

%Lateral connections
%Precalc angle differences
angdiffs = zeros(trials, 'single');
for a = 1:trials
    for b = a+1:trials
        angle_a = a_min + (a-1)*a_interval;
        angle_b = a_min + (b-1)*a_interval;
        angle_diff = abs(angle_a - angle_b);
        if angle_diff > pi/2
            angle_diff = pi - angle_diff;
        end
        angdiffs(a,b) = angle_diff;
        angdiffs(b,a) = angle_diff;
    end
end

%Experimental - lateral excitation and inhibition on GPU
%Only use with GPU mode enabled
if nlat > 0 && (lat_inhibfac ~= 0 || lat_excitefac ~= 0)
    for lateral = 1:nlat
        imgsize = size(large_img);
        pagree = libpointer('singlePtr', zeros(imgsize, 'single'));
        pdisagree = libpointer('singlePtr', zeros(imgsize, 'single'));
        pres = libpointer('singlePtr', best_activ);
        panglei = libpointer('int32Ptr', best_ai);
        panglediff = libpointer('singlePtr', angdiffs);
        pfilt = libpointer('singlePtr', filters);
        
        calllib('MultiConvolveFFT2D_CUDA', 'FindAgreement2D', ...
            pagree, pdisagree, pfilt, ...
            size(filters, 2), size(filters, 1), size(filters, 3), ...
            panglediff, pres, panglei, ...
            imgsize(1), imgsize(2), latmaxang);
        %imgsize(1), imgsize(2), pi);
        
        clear pres panglei ptubei panglediff pfilt;
        agree_result = get(pagree, 'Value');
        disagree_result = get(pdisagree, 'Value');
        clear pagree pdisagree;
        
        best_activ = best_activ + ...
            (lat_excitefac .* agree_result) + ...
            (lat_inhibfac .* disagree_result);
        
        %Normalise back to sensible range
        range_factor = 4;
        best_activ = (best_activ - mean(best_activ(:))) ./ (std(best_activ(:))/range_factor);
        best_activ = best_activ .* ba_sd0 + ba_mean;
        ba_sd = std(best_activ(:));
    end
end

%unloadlibrary('MultiConvolveFFT2D_CUDA');


%Distance weighting function
[xg yg] = meshgrid((-r:r)*ds);
%Basic euclidean
dist_cost = 0.3*ba_sd*sqrt(xg.^2 + yg.^2);

%Location repeat cost
%repeat_cost = 50;
repeat_cost = repeat_costfac * 3 * ba_sd;

angle_cost = (exp((0:1/trials:1-1/trials)*5)-1) * angle_costfac * ba_sd / 500;

%Backwards distance cost
back_size = floor(stepsize/2);
[xg yg] = meshgrid((-back_size:back_size));
back_dist = sqrt(xg.^2 + yg.^2);
back_dist_cost = repeat_cost - back_dist;
back_dist_cost(back_dist>back_size) = 0;

%ba_thresh = ba_mean+cand_thresh*ba_sd;

%baf_mean = mean(best_activ_fuzz(:));
%baf_sd = std(best_activ_fuzz(:));
%baf_thresh = baf_mean+cand_thresh*baf_sd;

%fuzz_better = (best_activ-ba_mean)./ba_sd < (best_activ_fuzz-baf_mean)./baf_sd;
%imagesc([best_activ, best_activ_fuzz, fuzz_better.*100]);

if display_progress
    subplot(2,3,4);
    colormap(gray);
    imagesc([best_activ]);
    axis image off;
    set(gca,'Position',[0.01 0.01 0.32 0.49])
    xlim([1+r, wwwx-r]);
    ylim([1+r, wwwy-r]);
    pause(0.1);
end

%pause;

%Just take the top score
[order_ba order_ba_ix] = sort(best_activ(:), 'descend');

top_cand = 0;

line_points = zeros(wwwy,wwwx);
line_junctions = zeros(wwwy,wwwx);
line_split_junctions = zeros(wwwy,wwwx);
line_join_junctions = zeros(wwwy,wwwx);
line_ends = zeros(wwwy,wwwx);
line_ends_edge = zeros(wwwy,wwwx);
line_ends_repeat = zeros(wwwy,wwwx);
line_ends_lost = zeros(wwwy,wwwx);
line_ends_juncy = zeros(wwwy,wwwx);

line_mito = zeros(wwwy,wwwx);
line_syn = zeros(wwwy,wwwx);

%This is where the central membrane is
%TODO: Thickness!
%target_thickness = best_thickness(i);

%mem_lines = cell(candi,1);
%score, x, y, xoffset, yoffset, angle, angleindex, next_direction.

line0 = struct('segment', {}, 'next_angle', {}, 'next_angleindex', {}, 'next_direction', {}, 'stillgoing', {});
%         line0 = struct( 'score', {}, ...
%             'x', {}, 'y', {}, 'xoffset', {}, 'yoffset', {}, ...
%             'angle', {}, 'angleindex', {}, 'next_direction', {}, ...
%             'junction_part', {}, 'stillgoing', {});

mem_lines = cell(0,1);

for step = 1:nsteps
    
    %Check if there any new junctions from the last run
    junctions = 0;
    for ml = 1:length(mem_lines)
        if all([mem_lines{ml}(:).stillgoing])
            junctions = 1;
            break;
        end
    end
    
    if ~junctions
        %Add a new start candidate
        top_cand = top_cand + 1;
        ok_cand = 0;
        while ~ok_cand
            [tcy,tcx] = ind2sub(size(best_activ),order_ba_ix(top_cand));
            %Check for too close to edge
            if tcx > r && tcx < wwwx-r && tcy > r && tcy < wwwy-r
                %Check for too close to other lines
                junc_points = line_points( ...
                    tcy-startpoint_radius:tcy+startpoint_radius, ...
                    tcx-startpoint_radius:tcx+startpoint_radius);
                junc_points = junc_points.*startpoint_area;
                if ~any(junc_points)
                    ok_cand = 1;
                else
                    %disp('TC too bunchy');
                end
            else
                %disp('TC too edgy');
            end
            if ~ok_cand
                %no good - move to next candidate
                top_cand = top_cand + 1;
            end
        end
        
        if (best_activ(tcy, tcx)-ba_mean) / ba_sd < start_thresh
            %No more start points
            break;
        end
        
        if verbose
            fprintf(1,'-----New Start Point-----\nDown to top %d, score %2.2f, sd %1.2f.\n', ...
                top_cand, best_activ(tcy, tcx), ...
                (best_activ(tcy, tcx)-ba_mean) / ba_sd);
        end
        
        ml = length(mem_lines) + 2;
        
        %Add the top candidate as the start of a line
        [tcy,tcx] = ind2sub(size(best_activ),order_ba_ix(top_cand));
        candx = tcx;
        candy = tcy;
        %Sanity Check
        if segment_index.getSegment(tcx, tcy, zi) ~= 0
            fprintf(1,'WARNING: Duplicate start point - this shouldn''t happen.\n');
        end
        start_seg = segment_index.newSegment( ...
            tcx, tcy, zi, 0, 0, ...
            a_min + (best_ai(tcy, tcx)-1)*a_interval, ...
            best_ai(tcy, tcx), best_activ(tcy, tcx), false);
        
        thisline = line0;
        thisline(1).segment = start_seg;
        thisline(1).next_angle = start_seg.angle;
        thisline(1).next_angleindex = start_seg.angleindex;
        thisline(1).next_direction = 1;
        thisline(1).stillgoing = true;
        mem_lines{ml} = thisline;
        
        %Repeat in the reverse direction
        mem_lines{ml-1} = thisline;
        mem_lines{ml-1}(1).next_direction = -1;
        
        line_points(tcy, tcx) = 1;
        
    end;
    
    new_line_seg = 0;
    %Extend each line as far as possible
    for linen = 1:length(mem_lines)
        thisline = mem_lines{linen};
        lend = length(thisline);
        %Extend the line by one step
        while thisline(lend).stillgoing
            
            current_seg = thisline(lend).segment;
            
            %Do the next step
            target_xy = [current_seg.x current_seg.y];
            target_angle = thisline(lend).next_angle;
            target_direction = thisline(lend).next_direction;
            
            target_offset_x = target_direction * round(stepsize*sin(target_angle));
            target_offset_y = target_direction * -round(stepsize*cos(target_angle));
            
            best = struct('score', -Inf);
            
            index_y = (target_xy(2)-r:target_xy(2)+r) + target_offset_y;
            index_x = (target_xy(1)-r:target_xy(1)+r) + target_offset_x;
            last_y = r+1-target_offset_y;
            last_x = r+1-target_offset_x;
            if any(index_y<1) || any(index_y>wwwy) || ...
                    any(index_x<1) || any(index_x>wwwx)
                %This line has met an edge
                if verbose
                    fprintf(1,'Edge at (%d, %d)\n', target_xy(1), target_xy(2));
                end
                thisline(lend).stillgoing = false;
                line_ends(current_seg.y, current_seg.x) = 1;
                line_ends_edge(current_seg.y, current_seg.x) = 1;
                mem_lines{linen} = thisline;
                continue;
            end
            
            %Check for mitochondria or synapse
            target_img = large_img( ...
                (target_xy(2)-r:target_xy(2)+r) + target_offset_y, ...
                (target_xy(1)-r:target_xy(1)+r) + target_offset_x);
            
            %Score by sum squared error
            %TODO random decision tree classifier or similar...
            mitoscore1 = sum(sum(target_img .* class_mito(:,:,current_seg.angleindex)));
            mitoscore2 = sum(sum(target_img .* class_mito(:,:,current_seg.angleindex + trials)));
            synscore1 = sum(sum(target_img .* class_syn(:,:,current_seg.angleindex)));
            synscore2 = sum(sum(target_img .* class_syn(:,:,current_seg.angleindex + trials)));
            
            synmax = max([synscore1 synscore2]);
            mitomax = max([mitoscore1 mitoscore2]);
            if synmax > mitomax && synmax > synthresh
                line_syn(target_xy(2), target_xy(1)) = 1;
                current_seg.type_syn = true;
                if verbose
                    fprintf(1,'Synscore:%d\n',synmax);
                end
            elseif mitomax > synmax && mitomax > mitothresh
                line_mito(target_xy(2), target_xy(1)) = 1;
                current_seg.type_mito = true;
                if verbose
                    fprintf(1,'Mitoscore:%d\n',mitomax);
                end
            end
            
            %            ascores = zeros(trials,1);
            %
            %             for a = 1:trials
            %
            %                 angle = a_min + (a-1)*a_interval;
            %                 angle_diff = abs(angle - thisline(lend).next_angle);
            %                 if angle_diff > pi/2
            %                     angle_diff = pi - angle_diff;
            %                 end
            %                 angle_diff_i = round(angle_diff/a_interval) + 1;
            %
            %                 %                     g = filters(:,:,a);
            %                 %
            %                 %                     %Convolve this around the large image to see where the best candidate
            %                 %                     %location might be.
            %                 %                     %Subtract distance cost and angle cost
            %                 %                     scores = conv2(target_img, g, 'same') - dist_cost - angle_cost(angle_diff_i);
            %                 %Use filterset already calculated
            %                 raw_score = activ(index_y, index_x, a);
            %                 %Subtract distance cost and angle cost
            %                 scores = raw_score - dist_cost - angle_cost(angle_diff_i);
            %                 %inhibit the previous location
            %                 backstep_y = last_y-back_size:last_y+back_size;
            %                 backstep_x = last_x-back_size:last_x+back_size;
            %                 scores(backstep_y,backstep_x) = ...
            %                     scores(backstep_y,backstep_x) - back_dist_cost;
            %                 %imagesc(scores);pause;
            %
            %                 [b ix] = max(scores(:));
            %                 ascores(a) = b + angle_cost(angle_diff_i);
            %
            %                 if (b > best.score)
            %                     %This is the new best score
            %                     %Find the x and y location
            %                     by = mod(ix, w);
            %                     if by == 0
            %                         by = w;
            %                     end
            %                     bx = floor(ix/w)+1;
            %                     %Record best x y (offset) and angle
            %                     best = struct( ...
            %                         'score', b, ...
            %                         'x', bx, ...
            %                         'y', by, ...
            %                         'movex', bx-(r+1)+target_offset_x, ...
            %                         'movey', by-(r+1)+target_offset_y, ...
            %                         'realx', target_xy(1) + bx - (r+1) + target_offset_x, ...
            %                         'realy', target_xy(2) + by - (r+1) + target_offset_y, ...
            %                         'angle', angle, ...
            %                         'angleindex', a);
            %                     best_scores = scores;
            %                 end
            %             end
            
            
            anglei = best_ai(index_y, index_x);
            angle = a_min + (anglei-1)*a_interval;
            angle_diff = abs(angle - thisline(lend).next_angle);
            angle_diff(angle_diff > pi/2) = pi - angle_diff(angle_diff > pi/2);
            angle_diff_i = round(angle_diff/a_interval) + 1;
            
            %Use filterset already calculated
            raw_score = best_activ(index_y, index_x);
            %Subtract distance cost and angle cost
            scores = raw_score - dist_cost - angle_cost(angle_diff_i);
            %inhibit the previous location
            backstep_y = last_y-back_size:last_y+back_size;
            backstep_x = last_x-back_size:last_x+back_size;
            scores(backstep_y,backstep_x) = ...
                scores(backstep_y,backstep_x) - back_dist_cost;
            %imagesc(scores);pause;
            
            [b ix] = max(scores(:));
            
            if (b > best.score)
                %This is the new best score
                %Find the x and y location
                by = mod(ix, w);
                if by == 0
                    by = w;
                end
                bx = floor(ix/w)+1;
                %Record best x y (offset) and angle
                best = struct( ...
                    'score', b, ...
                    'x', bx, ...
                    'y', by, ...
                    'movex', bx-(r+1)+target_offset_x, ...
                    'movey', by-(r+1)+target_offset_y, ...
                    'realx', target_xy(1) + bx - (r+1) + target_offset_x, ...
                    'realy', target_xy(2) + by - (r+1) + target_offset_y, ...
                    'angle', angle(ix), ...
                    'angleindex', anglei(ix));
                best_scores = scores;
            end
            
            if display_working
                target_img = large_img( ...
                    (target_xy(2)-r:target_xy(2)+r) + target_offset_y, ...
                    (target_xy(1)-r:target_xy(1)+r) + target_offset_x);
                figure(2);
                %Display the working for this step
                colormap(gray);
                subplot(3,3,1);
                imagesc(target_img);
                axis image off;
                subplot(3,3,2);
                imagesc(filters(:,:,round((target_angle - a_min) / a_interval + 1)));
                axis image off;
                subplot(3,3,3);
                imagesc(dist_cost);
                axis image off;
                subplot(3,3,4);
                %Mark the chosen location
                target_img(best.y,:) = target_img(best.y,:) + 0.5;
                target_img(:,best.x) = target_img(:,best.x) + 0.5;
                imagesc(target_img);
                axis image off;
                subplot(3,3,5);
                imagesc(best_scores);
                axis image off;
                subplot(3,3,6);
                imagesc(filters(:,:,best.angleindex));
                axis image off;
                pause;
            end
            
            %Now we have the best candidate location
            %Link to the next one
            next_xy = [best.realx best.realy];
            
            if abs(thisline(lend).next_angle-best.angle) > pi/2
                %Angle flip
                next_direction = -target_direction;
            else
                next_direction = target_direction;
            end
            
            %             junc_points = line_points( ...
            %                 next_xy(2)-junction_radius:next_xy(2)+junction_radius, ...
            %                 next_xy(1)-junction_radius:next_xy(1)+junction_radius);
            junc_points = get_neighbours_pad(line_points, next_xy(2), next_xy(1), junction_radius);
            junc_points = junc_points.*junction_area;
            
            if (next_xy(1) == current_seg.x && next_xy(2) == current_seg.y)
                %Repeat location - end this line
                thisline(lend).stillgoing = false;
                line_ends(current_seg.y, current_seg.x) = 1;
                line_ends_repeat(current_seg.y, current_seg.x) = 1;
                mem_lines{linen} = thisline;
                if verbose
                    fprintf(1,'Repeat at (%d, %d).\n', next_xy(1), next_xy(2));
                end
            elseif best.score < ba_mean + stop_thresh*ba_sd
                %Score too low - stop here
                thisline(lend).stillgoing = false;
                line_ends(current_seg.y, current_seg.x) = 1;
                line_ends_lost(current_seg.y, current_seg.x) = 1;
                mem_lines{linen} = thisline;
                if verbose
                    fprintf(1,'Stopped (score too low) at (%d,%d).\n', ...
                        next_xy(1), next_xy(2));
                end
            elseif any(junc_points(:)) && lend >= junction_nmax && ...
                    all([thisline(lend-junction_nmax+1:lend).segment.juncpart])
                %Too many junctions in a row - stop here
                thisline(lend).stillgoing = false;
                line_ends(current_seg.y, current_seg.x) = 1;
                line_ends_juncy(current_seg.y, current_seg.x) = 1;
                mem_lines{linen} = thisline;
                if verbose
                    fprintf(1,'Stopped (too many junctions) at (%d,%d).\n', ...
                        next_xy(1), next_xy(2));
                end
            else
                
                %Check for existing segment
                next_seg = segment_index.getSegment(next_xy(1), next_xy(2), zi);
                if next_seg ~= 0
                    %Use this segment and make it a junction
                    next_seg.juncpart = true;
                    line_join_junctions(next_seg.y, next_seg.x) = 1;
                    line_junctions(next_seg.y, next_seg.x) = 1;
                    if verbose
                        fprintf(1,'Made one join junction at (%d,%d).\n', ...
                            next_seg.x, next_seg.y);
                    end
                else
                    %Add a new segment
                    next_seg = segment_index.newSegment( ...
                        next_xy(1), next_xy(2), zi, ...
                        best.movex, best.movey, best.angle, best.angleindex, ...
                        best.score, false);
                end
                
                if current_seg.next == 0 && next_seg.prev == 0
                    current_seg.setNext(next_seg);
                elseif next_seg.prev == 0
                    %Split junction completion - just link the previous
                    next_seg.prev = current_seg;
                    next_seg.linkBoth(current_seg);
                    
                    current_seg.juncpart = true;
                    line_split_junctions(current_seg.y, current_seg.x) = 1;
                    line_junctions(current_seg.y, current_seg.x) = 1;
                    if verbose
                        fprintf(1,'Completed a split junction (or start point) at (%d,%d).\n', current_seg.x, current_seg.y);
                    end
                elseif current_seg.next == 0
                    %Existing segment - link to it
                    current_seg.next = next_seg;
                    next_seg.linkBoth(current_seg);
                    %Continue as normal (angle might be different this time)
                    %fprintf(1,'Found a duplicate point at (%d,%d).\n', next_seg.x, next_seg.y);
                else
                    %Both linked already
                    %Must be a junction completion and existing point???
                    next_seg.linkBoth(current_seg);
                    
                    current_seg.juncpart = true;
                    line_split_junctions(current_seg.y, current_seg.x) = 1;
                    line_junctions(current_seg.y, current_seg.x) = 1;
                    if verbose
                        fprintf(1,'Completed a split junction (or start point) AND found a duplicate point at (%d,%d).\n', current_seg.x, current_seg.y);
                    end
                end
                
                lnew = lend+1;
                thisline(lnew).segment = next_seg;
                thisline(lnew).next_angle = next_seg.angle;
                thisline(lnew).next_angleindex = next_seg.angleindex;
                thisline(lnew).next_direction = next_direction;
                thisline(lnew).stillgoing = true;
                
                %Check for join junctions
                if any(junc_points(:))
                    %Point nearby - check for junction
                    [junc_y junc_x] = find(junc_points);
                    for junci = 1:length(junc_y)
                        junc_seg = segment_index.getSegment( ...
                            next_xy(1) + junc_x(junci) - junction_radius - 1, ...
                            next_xy(2) + junc_y(junci) - junction_radius - 1, ...
                            zi);
                        if junc_seg ~= next_seg && ...
                                junc_seg ~= next_seg.prev
                            
                            next_seg.linkBoth(junc_seg);
                            next_seg.juncpart = true;
                            %junc_seg.juncpart = true;
                            
                            %Mark a join junction at the current point
                            line_join_junctions(next_xy(2), next_xy(1)) = 1;
                            line_junctions(next_xy(2), next_xy(1)) = 1;
                            if verbose
                                fprintf(1,'Made one join junction at (%d,%d).\n', ...
                                    next_seg.x, next_seg.y);
                            end
                        end
                    end
                else
                    %Check for split junctions
                    %Do not split if this is already a junction
                    %Do not split from the -1 direction if this is a seed
                    %Do not attempt to split if we are too near an edge
                    if ~current_seg.juncpart ...
                            && ~(lend==1 && thisline(lend).next_direction == -1) ...
                            && ~(next_seg.x+stepsize > wwwx || ...
                            next_seg.x-stepsize <= 0 || ...
                            next_seg.y+stepsize > wwwy || ...
                            next_seg.y-stepsize <= 0)
                        
                        %Angles to check
                        junc_check_d = [1:trials, 1:trials; ones(1,trials), -ones(1,trials)];
                        
                        %Don't look where we are already going
                        current_ix = thisline(lnew).next_angleindex + ...
                            trials*(thisline(lnew).next_direction==-1);
                        %Don't look where we have already been
                        prev_ix = thisline(lnew).next_angleindex + ...
                            trials*(thisline(lend).next_direction==1);
                        
                        ignore_ix = [current_ix-min_junction_a:current_ix+min_junction_a, ...
                            prev_ix-min_junction_a:prev_ix+min_junction_a];
                        
                        ignore_ix(ignore_ix>2*trials) = ignore_ix(ignore_ix>2*trials) - 2*trials;
                        ignore_ix(ignore_ix<1) = ignore_ix(ignore_ix<1) + 2*trials;
                        
                        junc_check_d(1,ignore_ix) = 0;
                        junc_check_d = junc_check_d(:,junc_check_d(1,:)~=0);
                        
                        jangle = a_min + (junc_check_d(1,:)-1) .* a_interval;
                        junc_offset_x = junc_check_d(2,:) .* round(stepsize*sin(jangle));
                        junc_offset_y = junc_check_d(2,:) .* -round(stepsize*cos(jangle));
                        %                         jind = sub2ind(size(activ), ...
                        %                             target_xy(2) + best.movey + junc_offset_y, ...
                        %                             target_xy(1) + best.movex + junc_offset_x, ...
                        %                             mod(junc_check_d(1,:)-1,trials)+1);
                        %                        jscore = activ(jind);
                        jind = sub2ind(size(best_activ), ...
                            target_xy(2) + best.movey + junc_offset_y, ...
                            target_xy(1) + best.movex + junc_offset_x);
                        jscore = best_activ(jind);
                        jscore_max = imregionalmax(jscore);
                        
                        %Check edges
                        if jscore_max(1) && jscore(length(jscore)) > jscore(1)
                            jscore_max(1) = 0;
                        elseif jscore_max(1) && jscore(length(jscore)) == jscore(1)
                            jscore_max(1) = jscore(length(jscore));
                        end
                        if jscore_max(length(jscore)) && jscore(length(jscore)) < jscore(1)
                            jscore_max(length(jscore)) = 0;
                        elseif jscore_max(length(jscore)) && jscore(length(jscore)) == jscore(1)
                            jscore_max(length(jscore)) = jscore(1);
                        end
                        
                        maxangles = (jscore.*jscore_max) > (ba_mean + start_junction_thresh*ba_sd);
                        candj = find(maxangles);
                        %Remove if too close to others
                        cjx = 1;
                        while cjx <= length(candj)
                            thisc = junc_check_d(:,candj(cjx));
                            thisl =  thisc(1) + trials.*(thisc(2)==-1);
                            others = candj(candj~=candj(cjx));
                            otherc = junc_check_d(:,others);
                            dist = abs(thisl - (otherc(1,:) + trials.*(otherc(2,:)==-1)));
                            if any(dist < min_junction_a)
                                %Remove the lowest candidate
                                near = [candj(cjx) others(dist < min_junction_a)];
                                nearscores = jscore(near);
                                [minscore removeix] = min(nearscores);
                                candj = candj(candj~=near(removeix));
                            else
                                cjx = cjx + 1;
                            end
                        end
                        
                        for jindex = candj
                            
                            %Create a junction here
                            junc_seg = thisline(lnew).segment;
                            %junc_seg.juncpart = true;
                            newline = thisline(lnew);
                            newline(1).segment = junc_seg;
                            newline(1).next_angle = jangle(jindex);
                            newline(1).next_angleindex = junc_check_d(1,jindex);
                            newline(1).next_direction = junc_check_d(2,jindex);
                            newline(1).stillgoing = true;
                            
                            lml = length(mem_lines);
                            mem_lines{lml+1} = newline;
                            
                            %thisline(lnew).junction_part = true;
                            %mem_lines{linen} = thisline;
                            
                            %line_split_junctions(junc_seg.y, junc_seg.x) = 1;
                            %line_junctions(junc_seg.y, junc_seg.x) = 1;
                            if verbose
                                fprintf(1,'Started a split junction at (%d,%d).\n', junc_seg.x, junc_seg.y);
                            end
                            
                            if display_junc_working
                                fprintf(1,'Cjunk next %d(%d), prev %d(%d), junc %d(%d).', ...
                                    thisline(lend).next_angleindex, thisline(lend).next_direction, ...
                                    thisline(lnew).next_angleindex, thisline(lnew).next_direction, ...
                                    newline(1).next_angleindex, newline(1).next_direction);
                                target_img = large_img( ...
                                    (junc_seg.y-r:junc_seg.y+r), ...
                                    (junc_seg.x-r:junc_seg.x+r));
                                figure(2);
                                colormap(gray);
                                subplot(3,3,1);
                                imagesc(target_img);
                                hold on;
                                plot(junc_offset_x(jindex)+r-1, junc_offset_y(jindex)+r-1, 'bo', 'LineWidth', 2, 'MarkerSize', 10);
                                hold off;
                                axis image off;
                                subplot(3,3,2);
                                imagesc(filters(:,:,best.angleindex));
                                axis image off;
                                subplot(3,3,3);
                                imagesc(dist_cost);
                                axis image off;
                                subplot(3,3,4);
                                %                                 imagesc(activ( ...
                                %                                     (junc_seg.y-r:junc_seg.y+r), ...
                                %                                     (junc_seg.x-r:junc_seg.x+r), ...
                                %                                     newline(1).next_angleindex));
                                imagesc(best_activ( ...
                                    (junc_seg.y-r:junc_seg.y+r), ...
                                    (junc_seg.x-r:junc_seg.x+r)));
                                subplot(3,3,5);
                                imagesc(filters(:,:,newline(1).next_angleindex));
                                axis image off;
                                pause;
                            end;
                            
                        end
                    end
                end
                
                line_points(next_seg.y, next_seg.x) = 1;
                if next_xy(1) <= r || next_xy(1) >= wwwx-r || ...
                        next_xy(2) <= r || next_xy(2) >= wwwy-r
                    %Too close to edge - end the line
                    thisline(lnew).stillgoing = false;
                    line_ends(next_seg.y, next_seg.x) = 1;
                    line_ends_edge(next_seg.y, next_seg.x) = 1;
                    if verbose
                        fprintf(1,'Edge at (%d, %d)\n', next_seg.x, next_seg.y);
                    end
                else
                    new_line_seg = new_line_seg+1;
                end
                mem_lines{linen} = thisline;
                lend = length(thisline);
            end
        end
    end
    
    if display_progress && (mod(step,20)==0)
        %Display progress
        leg = {'Membrane'};
        %figure(1);
        set(0,'CurrentFigure',1)
        subplot(2,3,[2 3 5 6]);
        colormap(gray);
        imagesc(large_img);
        axis image off;
        set(gca,'Position',[0.34 0.01 0.65 0.99])
        xlim([1+r, wwwx-r]);
        ylim([1+r, wwwy-r]);
        hold on;
        %Start points
        %plot(candx, candy, 'cx', 'LineWidth', 2, 'MarkerSize',10);
        %End points
        %[endy endx] = find(line_ends);
        %plot(endx, endy, 'ms', 'LineWidth', 2, 'MarkerSize',10);
        %[endy endx] = find(line_ends_repeat);
        %plot(endx, endy, 'ks', 'LineWidth', 2, 'MarkerSize',10);
        %[endy endx] = find(line_ends_lost);
        %plot(endx, endy, 'ys', 'LineWidth', 2, 'MarkerSize',10);
        %[endy endx] = find(line_ends_juncy);
        %plot(endx, endy, 'ws', 'LineWidth', 2, 'MarkerSize',10);
        %[endy endx] = find(line_ends_edge);
        %plot(endx, endy, 'ms', 'LineWidth', 2, 'MarkerSize',10);
        %Junctions
        %[endy endx] = find(line_junctions);
        %plot(endx, endy, 'bo');
        %[endy endx] = find(line_split_junctions);
        %plot(endx, endy, 'bo', 'LineWidth', 2, 'MarkerSize',10);
        %[endy endx] = find(line_join_junctions);
        %plot(endx, endy, 'wo', 'LineWidth', 2, 'MarkerSize',10);
        %Classes
        [endy endx] = find(line_mito);
        if ~isempty(endy)
            plot(endx, endy, 'ko', 'LineWidth', 2, 'MarkerSize',12);
            leg = {'Organelle', leg{:}};
        end
        [endy endx] = find(line_syn);
        if ~isempty(endy)
            plot(endx, endy, 'ro', 'LineWidth', 2, 'MarkerSize',12);
            leg = {'Synapse', leg{:}};
        end
        %Lines
        min_best = 100;
        for linen = 1:length(mem_lines)
            segs = [mem_lines{linen}(:).segment];
            linex = [segs.x];
            liney = [segs.y];
            plot(linex, liney, 'gx-', 'LineWidth', 2);
            min_best = min(min_best, min([segs.score]));
        end
        hold off;
        
        %legend(leg, 'Location', 'SouthEast', 'Color', [0.7 0.7 0.7]);
        
        if verbose
            fprintf(1,'Min best score of %d.', min_best);
            fprintf(1,'Finished step %d with %d new line segments.\n', step, new_line_seg);
        else
            fprintf(1,'Finished step %d.\n', step);
        end
        
        pause(0.01);
    end
    
end

if display_progress && step > 1
    %Display progress
    leg = {'Membrane'};
    %figure(1);
    set(0,'CurrentFigure',1)
    subplot(2,3,[2 3 5 6]);
    colormap(gray);
    imagesc(large_img);
    axis image off;
    set(gca,'Position',[0.34 0.01 0.65 0.99])
    xlim([1+r, wwwx-r]);
    ylim([1+r, wwwy-r]);
    hold on;
    %Start points
    %plot(candx, candy, 'cx', 'LineWidth', 2, 'MarkerSize',10);
    %End points
    %[endy endx] = find(line_ends);
    %plot(endx, endy, 'ms', 'LineWidth', 2, 'MarkerSize',10);
    %[endy endx] = find(line_ends_repeat);
    %plot(endx, endy, 'ks', 'LineWidth', 2, 'MarkerSize',10);
    %[endy endx] = find(line_ends_lost);
    %plot(endx, endy, 'ys', 'LineWidth', 2, 'MarkerSize',10);
    %[endy endx] = find(line_ends_juncy);
    %plot(endx, endy, 'ws', 'LineWidth', 2, 'MarkerSize',10);
    %[endy endx] = find(line_ends_edge);
    %plot(endx, endy, 'ms', 'LineWidth', 2, 'MarkerSize',10);
    %Junctions
    %[endy endx] = find(line_junctions);
    %plot(endx, endy, 'bo');
    %[endy endx] = find(line_split_junctions);
    %plot(endx, endy, 'bo', 'LineWidth', 2, 'MarkerSize',10);
    %[endy endx] = find(line_join_junctions);
    %plot(endx, endy, 'wo', 'LineWidth', 2, 'MarkerSize',10);
    %Classes
    [endy endx] = find(line_mito);
    if ~isempty(endy)
        plot(endx, endy, 'ko', 'LineWidth', 2, 'MarkerSize',12);
        leg = {'Organelle', leg{:}};
    end
    [endy endx] = find(line_syn);
    if ~isempty(endy)
        plot(endx, endy, 'ro', 'LineWidth', 2, 'MarkerSize',12);
        leg = {'Synapse', leg{:}};
    end
    %Lines
    min_best = 100;
    for linen = 1:length(mem_lines)
        segs = [mem_lines{linen}(:).segment];
        linex = [segs.x];
        liney = [segs.y];
        plot(linex, liney, 'gx-', 'LineWidth', 2);
        min_best = min(min_best, min([segs.score]));
    end
    hold off;
    
    %legend(leg, 'Location', 'SouthEast', 'Color', [0.7 0.7 0.7]);
    
    if verbose
        fprintf(1,'Min best score of %d.', min_best);
        fprintf(1,'Finished step %d with %d new line segments.\n', step, new_line_seg);
    else
        fprintf(1,'Finished step %d.\n', step);
    end
    
    pause(0.01);
end

end
