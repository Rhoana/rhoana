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


function [fmem ffuzz fall class_syn class_mito] = generate_filters_nn_g_r(ds, out_radius, nangles, ncurves, gamma)
%Generate filters for membrane, mitochondria, synapse and fuzzy membrane.

bias = -1;
outw = out_radius*2+1;

gamma = round(gamma*100)/100;
filename = sprintf('data\\5filters_nn_g_w%d_ds%1.2f_angles%d_curves%d_gamma%1.2f.mat', outw, ds, nangles, ncurves, gamma);

if exist(filename,'file')
    load(filename);
    %clf;
%     imagesc([fmem(:,:,1), ffuzz(:,:,1), fall(:,:,1); ...
%         class_syn(:,:,1), class_mito(:,:,1), zeros(size(fall(:,:,1))); ...
%         class_syn(:,:,1 + nangles), class_mito(:,:,1 + nangles), zeros(size(fall(:,:,1))); ...
%         ]);
    %imagesc([fmem(:,:,1), ffuzz(:,:,1); ...
    %    class_syn(:,:,1), class_mito(:,:,1)]);
    %axis image off;
    %pause;
    return;
end

%Larger radius required for rotation (rounded to nearest 5 pixles)
r = ceil(sqrt(2*(out_radius^2))/5)*5;
w = r*2+1;
zw = 3;

a_min = 0;
a_max = pi;
a_interval = (a_max-a_min) / nangles;

c_min = 0;
c_max = 1;
c_interval = (c_max-c_min) / ncurves;

%Load or learn membrane properties
[mem_rf, fuzz_rf, all_rf, syn_rf, mito_rf] = learn_membrane_step(ds, w, bias, 1);

fprintf(1, 'Calibrating filters.\n');

%Normalise
mem_rf = mem_rf ./ max(abs(mem_rf(:)));
fuzz_rf = fuzz_rf ./ max(abs(fuzz_rf(:)));
all_rf = all_rf ./ max(abs(all_rf(:)));
syn_rf = syn_rf ./ max(abs(syn_rf(:)));
mito_rf = mito_rf ./ max(abs(mito_rf(:)));

fmem = zeros(outw,outw,nangles*ncurves);
fall = zeros(outw,outw,nangles*ncurves);
ffuzz = zeros(outw,outw,nangles*ncurves);
class_mito = zeros(outw,outw,nangles*2*ncurves);
class_syn = zeros(outw,outw,nangles*2*ncurves);

rad2deg = 360/(2*pi);

for a = 1:nangles
    angle = a_min + (a-1)*a_interval;
    for c = 1:ncurves
        curve = c_min + (c-1)*c_interval;
        fmem(:,:,a + nangles*(c-1)) = calibrate(mem_rf, ds, r, out_radius, angle, curve, gamma);
        ffuzz(:,:,a + nangles*(c-1)) = calibrate(fuzz_rf, ds, r, out_radius, angle, curve, gamma);
        fall(:,:,a + nangles*(c-1)) = calibrate(all_rf, ds, r, out_radius, angle, curve, gamma);
    end
    
    class = imrotate(mito_rf, -angle*rad2deg, 'bilinear', 'crop');
    class_mito(:,:,a) = class(r+1-out_radius:r+1+out_radius, r+1-out_radius:r+1+out_radius);
    class_mito(:,:,a+nangles) = class(r+1+out_radius:-1:r+1-out_radius, r+1+out_radius:-1:r+1-out_radius);

    class = imrotate(syn_rf, -angle*rad2deg, 'bilinear', 'crop');
    class_syn(:,:,a) = class(r+1-out_radius:r+1+out_radius, r+1-out_radius:r+1+out_radius);
    class_syn(:,:,a+nangles) = class(r+1+out_radius:-1:r+1-out_radius, r+1+out_radius:-1:r+1-out_radius);
    
end

%clf;
% imagesc([fmem(:,:,1), ffuzz(:,:,1), fall(:,:,1); ...
%     class_syn(:,:,1), class_mito(:,:,1), zeros(size(fall(:,:,1))); ...
%     class_syn(:,:,1 + nangles), class_mito(:,:,1 + nangles), zeros(size(fall(:,:,1))); ...
%     ]);
%pause;

%Save the filters
save(filename, 'fmem', 'ffuzz', 'fall', 'class_mito', 'class_syn');

end

function g3 = calibrate(ave, ds, r, out_radius, angle, curve, gamma)

%Default gamma
%gamma = 1;

%Default sigma
sigma = 10*ds;

%Calibrate g3 so that sum is close to 0
%Binary search
gmax = 20*ds;
gmin = 2*ds;
score_thresh = 1e-10;

g3 = creategabor3(ave, gmax, r, out_radius, gamma, angle, 1, curve);
score_max = sum(g3(:));
g3 = creategabor3(ave, gmin, r, out_radius, gamma, angle, 1, curve);
score_min = sum(g3(:));
new_score = score_min;

%Binary search
while abs(new_score) > score_thresh
    if sign(score_max) ~= sign(score_min)
        mid = (gmax+gmin)/2;
        g3 = creategabor3(ave, mid, r, out_radius, gamma, angle, 1, curve);
        new_score = sum(g3(:));
        if new_score > 0
            gmax = mid;
            score_max = new_score;
        else
            gmin = mid;
            score_min = new_score;
        end
        %fprintf(1,'.');
    else
        %disp('Error - couldn''t calibrate sigma - using default.');
        break;
    end
end

if abs(new_score) <= score_thresh
    sigma = mid;
    %fprintf(1,'Found best score at sigma=%f (%d).\n', sigma, new_score);
end

    g3 = creategabor3(ave, sigma, r, out_radius, gamma, angle, 1, curve);
%      imagesc(g3);
%      pause;

end
