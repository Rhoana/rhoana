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


function [mem_rf, fuzz_rf, all_rf, syn_rf, mito_rf] = learn_membrane_step(ds, w, bias, nsteps)

zw = 3;

filename = sprintf('data\\nnfilter2_multi_w%d_zw%d_b%1.2f.mat', w, zw, bias);

if exist(filename,'file')
    disp('Loading membrane properties.');
    load(filename);
else
    disp('Learning membrane properties.');
    
    %disp('Loading example files.');
    
    %Get membrane examples
    %Load the larger version (so we can convolve)
    load(sprintf('data\\loc_images_w%d_z%d_ds%1.2f.mat', w, zw, ds));
    %TODO: Use non-rotated images as -ve example?
    load(sprintf('data\\images_rot_w%d_z%d_ds%1.2f.mat', w, zw, ds));
    
    memi = locations(:,4) == 0;
    images_r_mem = expand((images_r(:,memi,2)-128)/128, w);
    
    %Get other examples
    load(sprintf('data\\loc_images_tm_w%d_z%d_ds%1.2f.mat', w, zw, ds));
    %TODO: Use non-rotated images as -ve example?
    load(sprintf('data\\images_tm_rot_w%d_z%d_ds%1.2f.mat', w, zw, ds));
    
    fuzzi = locations_tm(:,4) == 0;
    mitoi = locations_tm(:,4) == 1;
    syni = locations_tm(:,4) == 2;
    
    %Fuzzy membrane examples
    images_r_fuzz = expand((images_r_tm(:,fuzzi,2)-128)/128, w);
    
    %Mitochondron and synapse examples
    images_r_mito = (images_r_tm(:,mitoi,2)-128)/128;
    images_r_syn = (images_r_tm(:,syni,2)-128)/128;
    
    images_r_mito_y = expand_yonly(images_r_mito, w);
    %images_r_mito = expand(images_r_mito, w);
    
    images_r_syn_y = expand_yonly(images_r_syn, w);
    images_r_syn = expand(images_r_syn, w);
    
    %Get random examples
    load(sprintf('data\\loc_images_random_w%d_z%d_ds%1.2f.mat', w, zw, ds));
    
    images_r_rand = (images_rand(:,:,2)-128)/128;
    %Remove 137 and 344 (over the edge)
    images_r_rand = images_r_rand(:,[1:136,138:343,345:size(images_r_rand,2)]);
    %Just 100
    %images_r_rand = images_r_rand(:,1:100);
    images_r_rand = expand(images_r_rand, w);
    
    %Put together in learning matrix
    
    %disp('Learning membrane.');
    %Just membrane
    inputs = [images_r_mem, images_r_rand];
    targets = [repmat([1],1,size(images_r_mem,2)), ...
        repmat([0],1,size(images_r_rand,2))];
    tnn = adapt_one(inputs, targets, bias, nsteps);
    mem_rf = reshape(tnn.IW{1}(1,:), w, w);
    
    %disp('Learning fuzzy membrane.');
    %Just fuzzy
    inputs = [images_r_fuzz, images_r_rand];
    targets = [repmat([1],1,size(images_r_fuzz,2)), ...
        repmat([0],1,size(images_r_rand,2))];
    tnn = adapt_one(inputs, targets, bias);
    fuzz_rf = reshape(tnn.IW{1}(1,:), w, w);
    
    %disp('Learning membrane and fuzzy membrane.');
    %Membrane and fuzzy??
    %     inputs = [images_r_mem, images_r_fuzz, images_r_rand];
    %     targets = [repmat([1],1,size(images_r_mem,2)), ...
    %          repmat([1],1,size(images_r_fuzz,2)), ...
    %         repmat([0],1,size(images_r_rand,2))];
    
    %disp('Learning synapse membrane.');
    %Just synapse
    %     inputs = [images_r_syn, images_r_rand];
    %     targets = [repmat([1],1,size(images_r_syn,2)), ...
    %         repmat([0],1,size(images_r_rand,2))];
    
    %disp('Learning all membrane.');
    %All membrane classes in one
    inputs = [images_r_mem, images_r_fuzz, images_r_syn, images_r_rand];
    targets = [repmat([1],1,size(images_r_mem,2)), ...
        repmat([1],1,size(images_r_fuzz,2)), ...
        repmat([1],1,size(images_r_syn,2)), ...
        repmat([0],1,size(images_r_rand,2))];
    tnn = adapt_one(inputs, targets, bias);
    all_rf = reshape(tnn.IW{1}(1,:), w, w);
    
    %disp('Learning mitochondria / synapse classes.');
    %Synapse or Mitochondria, not membrane, not fuzzy
    inputs = [images_r_mem, images_r_fuzz, images_r_syn_y, images_r_mito_y, images_r_rand];
    targets = [repmat([0; 0],1,size(images_r_mem,2)), ...
        repmat([0; 0],1,size(images_r_fuzz,2)), ...
        repmat([1; 0],1,size(images_r_syn_y,2)), ...
        repmat([0; 1],1,size(images_r_mito_y,2)), ...
        repmat([0; 0],1,size(images_r_rand,2))];
    
    tnn = adapt_one(inputs, targets, bias);
    syn_rf = reshape(tnn.IW{1}(1,:), w, w);
    mito_rf = reshape(tnn.IW{1}(2,:), w, w);

    save(filename, 'mem_rf', 'fuzz_rf', 'all_rf', 'syn_rf', 'mito_rf');
    
end

% subplot(2,2,1);
% imagesc(mem_rf);
% axis image off;
% title('Membrane');
% set(gca, 'Position', [0.02 0.51 0.45 0.42]);
% 
% subplot(2,2,2);
% imagesc(fuzz_rf);
% axis image off;
% title('Fuzzy Membrane');
% set(gca, 'Position', [0.52 0.51 0.45 0.42]);
% 
% % subplot(2,3,3);
% % imagesc(all_rf);
% % axis image off;
% % title('All Membrane');
% 
% subplot(2,2,3);
% imagesc(syn_rf);
% axis image off;
% title('Synapse');
% set(gca, 'Position', [0.02 0.01 0.45 0.42]);
% 
% subplot(2,2,4);
% imagesc(mito_rf);
% axis image off;
% title('Mitochondria');
% set(gca, 'Position', [0.52 0.01 0.45 0.42]);
% 
% pause(0.1);
% %pause;

end

function nn = adapt_one(inputs, targets, bias, nsteps)
%Create and adapt a patter recognition neural network
nn = newpr(inputs, targets);
for gi = 1:size(nn.IW{1}, 1)
    nn.IW{1}(gi,:) = zeros(size(nn.IW{1}(gi,:)));
end
nn.b{1} = ones(size(nn.b{1})).*bias;
if nargin == 4
    nn.adaptParam.passes = nsteps;
else
    nn.adaptParam.passes = 1;
end
[nn,Y,E,Pf,Af] = adapt(nn,inputs,targets);
end
