function  [ i_sol_lbl_overlap, label_costs, i_sol_bin_overlap] = build_image_stack_from_fusion_solution( ...
                                                      clq_2D_win_ix, ...
                                                      clq_3D_win_ix, ...
                                                      clq_2D_data, ...
                                                      clq_3D_data, ...
                                                      i_stacks, ...
                                                      size_info, ...
                                                      flg_region_filling, ...
                                                      flg_crack_inpainting, ...
                                                      tess_smallest_seg_area_proportion, ...
                                                      link_costs)  

label_costs = [];
%% CHECK OUT ASSOCIATED UT file!!!

% This distance is not used at the moment (25 November 10.37 PM 2011)
% since we use tess_smallest_seg_area_proportion for the cleaning and 
% inpainting with distance = Inf. Still, we fix it to some value = 2.
dist_for_crack_inpainting  = 6;

new_empty_sppixels              = 'cc';
DEBUGGING                       = true;


struct2var(clq_2D_data.spt_maps);
struct2var(i_stacks);

nR            = size_info(1);
nC            = size_info(2);
n_frames      = size_info(3);

smallest_area_for_new_object = ...
   round((nR/tess_smallest_seg_area_proportion * nC/tess_smallest_seg_area_proportion));   

% INPUT: 
 % clq_3D_data
 % cl
% The next step is to figure out what is the CC label assigned to each
% of the winning 2D cliques and 3D links. To do this, we prepare a graph
% of winners. We can take clq_3D_non_cent for this
solution_graph     = logical(clq_3D_data.clq_3D_adj);
%% Discarding links from the solver:

% [link_As link_Bs ] = find(solution_graph); tic
link_As = clq_3D_data.links_As;
link_Bs = clq_3D_data.links_Bs;
% What links were discarded by the solver?
link_As(clq_3D_win_ix) = [];
link_Bs(clq_3D_win_ix) = [];
links_to_remove_lin_ix = sub2ind(size(solution_graph), link_As,link_Bs);
solution_graph(links_to_remove_lin_ix) = 0;

%% Connected components:                 
% If we have the bioinformatics toolbox we can also use 
% graphconncomp, but otherwise we should use matlab_bgl
% Remember thsi function finds the STRONGLY connected components:
% A DIRECTED graph is called STRONGLY connected if:
% "there is a path from each vertex in the graph to every other vertex"
solution_graph              = max(solution_graph, solution_graph');
[clqs_2D_labeling, sizes]   = components(solution_graph);                                                                                                                                                                                         
n_CCs = max(clqs_2D_labeling);   
%% Finding unique labels per CC'
% We want to output one label for each CC in the winning out in
% incremental order:
[unique_lbls_winners, ~, n]           = unique(clqs_2D_labeling(clq_2D_win_ix));
 % 'n' is very helpful!
 % clqs_2D_labeling(clq_2D_win_ix) = unique_lbls_winners(n);

% We assign a unique incremental label to the CCs of interest 
% (i.e. those marked by clq_2D_win_ix);
unique_inc_labels_winners = (1:numel(unique_lbls_winners))';
n_labels_on_winning_cliques = numel(unique_lbls_winners);

%% If all worked well...
% The next calls should output the same label consistently (lower for clqs_2D_re_labeling)
   %   winning_CC = 40;
   %   clqs_2D_labeling(   clqs_2D_labeling == unique_lbls_winners(winning_CC))
   %   clqs_2D_re_labeling(clqs_2D_labeling == unique_lbls_winners(winning_CC))

% The next calls would also output the same label consistently (lower for clqs_2D_re_labeling)
   %   winning_node = 40;
   %   clqs_2D_labeling(clqs_2D_labeling   == clqs_2D_labeling(clq_2D_win_ix(winning_node)))
   %   clqs_2D_re_labeling(clqs_2D_re_labeling == clqs_2D_re_labeling(clq_2D_win_ix(winning_node)))  
%%

%% Render solution from the connected components:
% INPUT:
 % clq_2D_data
 % i_stacks
 % winning 2D cliques
 % clqs_2D_labeling

clq_2D_win_ix = clq_2D_win_ix(:)';

i_sol_bin_overlap                = zeros(nR, nC, n_frames, 'uint32');
i_sol_lbl_overlap                = zeros(nR, nC, n_frames, 'uint32');
i_win_clq_2D_overlap             = zeros(nR, nC, n_frames, 'uint32');

i_s_bw_pixels_for_region_filling   = false(nR, nC, n_frames);
i_s_bw_pixel_cracks              = false(nR, nC, n_frames);
% Re-labeling to have incremental labels

n_winning_2D_cliques = numel(clq_2D_win_ix);
winner_label = zeros(n_winning_2D_cliques, 1);
for winner_ix = 1:n_winning_2D_cliques
   winner_label(winner_ix) = unique_inc_labels_winners( unique_lbls_winners == clqs_2D_labeling(clq_2D_win_ix(winner_ix)));
end


%% GET THE LABEL COSTS:
% For each label, find the cliques that have it
% Get the links between them
% Get the cost for the label
link_As = clq_3D_data.links_As;
link_Bs = clq_3D_data.links_Bs;

win_link_As = link_As(clq_3D_win_ix);
win_link_Bs = link_Bs(clq_3D_win_ix);
%%
label_costs = zeros(n_labels_on_winning_cliques, 1);
for this_label = 1:n_labels_on_winning_cliques
   winning_cliques_this_label = clq_2D_win_ix(find(winner_label == this_label));
   [r c] = find(clq_3D_data.clq_3D_adj(winning_cliques_this_label, winning_cliques_this_label));
   path_this_clique = [winning_cliques_this_label(r)' winning_cliques_this_label(c)'];
   
   win_link_As_on_the_path = ismember(win_link_As, path_this_clique);
   win_link_Bs_on_the_path = ismember(win_link_Bs, path_this_clique);

   
   link_costs_this_path    = sum((win_link_As_on_the_path & win_link_Bs_on_the_path) .* link_costs);
   label_costs(this_label)  = link_costs_this_path;   
end

%% Get the overlap of winning cliques
winner_ix = 0;

spt_mf_M = clq_2D_data.spt_mf_M;
i_s_L_sppst = i_stacks.i_s_L_sppst;

disp(['# of winning 2D cliques in this solution ' num2str(n_winning_2D_cliques)]);
disp('Building solution stack by assembling winning cliques one-by-one ...');
t_assembly = tic;

for clq_2D_ix = clq_2D_win_ix
  winner_ix = winner_ix +1;
   label_for_this_2D_clique = winner_label(winner_ix);     

   frame_ix    = clique_ix_to_frame_ix(clq_2D_ix);
   sppst_ixs_this_clique   = [find(spt_mf_M(clq_2D_ix,:))];
   i_L_superpst = squeeze(i_s_L_sppst(:,:,frame_ix));

   sppst_lbls_this_clique  = sppst_ix_to_sppst_lbl_ix(sppst_ixs_this_clique);   
   i_bw = uint32(logical(ismembc2(i_L_superpst, cast(sort(sppst_lbls_this_clique), class(i_L_superpst)))));
   i_sol_bin_overlap(:,:,frame_ix)  = i_sol_bin_overlap(:,:,frame_ix) + ...
                                     i_bw;

  i_sol_lbl_overlap(:,:,frame_ix)  = i_sol_lbl_overlap(:,:,frame_ix) + ...
                                     label_for_this_2D_clique * i_bw;
                                  
   i_win_clq_2D_overlap(:,:,frame_ix)  = i_win_clq_2D_overlap(:,:,frame_ix) + ...
                                             clq_2D_ix * i_bw;                                  
                                  
end

disp(['Done in ' secs2hms(toc(t_assembly))]);
%% Inpainting superpixels: (WE DON't DO THIS FOR NESTED FUSION!)
start_new_labels_from_value = max(i_sol_lbl_overlap(:)) + 1;

for frame_ix=1:n_frames
   i_L_tmp = squeeze(i_sol_lbl_overlap(:,:,frame_ix));

   
   i_L_zero = i_L_tmp == 0;
   [i_D , ~] = bwdist(~i_L_zero);

   % i_bw_pixel_cracks is not used at the moment (we are inpainting with
   % value = Inf)
   i_bw_pixels_cracks       = (i_D <= dist_for_crack_inpainting) & i_L_zero;
   i_s_bw_pixel_cracks(:,:,frame_ix) = i_bw_pixels_cracks;
   
   i_bw_pixels_not_cracks   = (i_D > dist_for_crack_inpainting) & i_L_zero;
   i_bw_pixels_not_cracks   = bwareaopen(i_bw_pixels_not_cracks, smallest_area_for_new_object);
   
   i_s_bw_pixels_for_region_filling(:,:,frame_ix) = i_bw_pixels_not_cracks;
%    i_inpainted = inpaint_superpixels(...
%          i_L_tmp, ...
%          start_new_labels_from_value, ...
%          new_empty_sppixels, ...
%          dist_for_crack_inpainting);
%       
%    i_sol_lbl_overlap(:,:,frame_ix)  = i_inpainted;
end


%% Add missing objects (remember, we can dilate i_s_bw)
CC = bwconncomp(i_s_bw_pixels_for_region_filling, 26);
i_s_add_L = uint32(labelmatrix(CC));
%%
i_s_add_L(i_s_add_L ~=0) = i_s_add_L(i_s_add_L ~=0) + (start_new_labels_from_value -1);

%% Add new connected components!????

if flg_region_filling
   i_s_L_new = i_sol_lbl_overlap + i_s_add_L;
else
   i_s_L_new = i_sol_lbl_overlap;
end

i_sol_lbl_overlap = i_s_L_new;
i_sol_lbl_overlap(i_s_bw_pixel_cracks) = 0;
%%
% This is just in case, although I checked and we don't usually have any
% missing boundaries!
%%
for frame_ix=1:n_frames
 i_sol_lbl_overlap(:,:,frame_ix) = fix_missing_boundaries(squeeze(i_sol_lbl_overlap(:,:,frame_ix)));
end

%% Crack inpainting:   
if flg_crack_inpainting
   for frame_ix=1:n_frames   
      i_L_tmp = sq(i_sol_lbl_overlap(:,:,frame_ix));
      if any(i_L_tmp(:))
         i_sol_lbl_overlap(:,:,frame_ix) =  inpaint_superpixels(i_L_tmp, Inf);
      else
         i_sol_lbl_overlap(:,:,frame_ix) = i_L_tmp;
      end
   end
end

% Return images at downsample resolution?
% Check out: from_i_thick_to_i_thin

%% DEBUGGING:

if DEBUGGING
   true;   
end
%%
%     cmap  = rand_cmap(i_sol_lbl_overlap);
%     n_CCs = max(i_sol_lbl_overlap(:));
%     
%     figure(fig_number);
%     clf;
%     for frame_ix = 1:n_frames
%       subaxis(3,ceil(n_frames/3),frame_ix)
%       imshow(squeeze(i_sol_lbl_overlap(:,:,frame_ix)), []);
%       colormap(cmap);
%       set(gca, 'CLim',[0 n_CCs]);
%       axis off;
%     end
% 
%     figure(fig_number+1);
%     clf;
%     ovlap_max = max(i_sol_bin_overlap(:));
% 
%     for frame_ix = 1:n_frames
%       subaxis(3,ceil(n_frames/3),frame_ix)
%       imshow(squeeze(i_sol_bin_overlap(:,:,frame_ix)), []);
%       colormap(cmap);
%       set(gca, 'CLim',[0 ovlap_max]);
%       axis off;
%     end


