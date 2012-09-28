function link_rep_data = get_3D_adjacency(clique_rep_data, i_stacks, adj_type, size_info)
try

nR = size_info(1);
nC = size_info(2);
n_frames = size_info(3);

% This is what this function needs:
  struct2var(clique_rep_data.spt_maps);
  n_total_cliques                = numel(clique_ix_to_frame_ix);
%   i_all_segments_not_centered = i_stacks.i_all_segments_not_centered;
%   i_all_segments_centered     = i_stacks.i_all_segments_centered;

%% Non-centered (to determine adjacency)

disp('Determining frame-to-frame adjacency from non-centered bitmaps');
tic_all_frames = tic;

spt_mf_M = clique_rep_data.spt_mf_M;

i_s_L_sppst = i_stacks.i_s_L_sppst;

rows = cell(n_frames-1,1);
cols = cell(n_frames-1,1);
vals = cell(n_frames-1,1);
disp(['Total # of cliques: ' num2str(n_total_cliques)]);
for frame_ix = 1:n_frames-1
   %%
   prev_spt_M = spt_mf_M(clique_ix_to_frame_ix == frame_ix, ...
                          sppst_ix_to_frame_ix ==  frame_ix);

   next_spt_M = spt_mf_M(clique_ix_to_frame_ix == frame_ix+1, ...
                       sppst_ix_to_frame_ix ==  frame_ix+1);

   prev_i_L_superpst = i_s_L_sppst(:,:,frame_ix);
   next_i_L_superpst = i_s_L_sppst(:,:,frame_ix+1);

   %% For getting the indexing in the adjacency matrices
   first_clq_prev = find(clique_ix_to_frame_ix == frame_ix, 1,'First');
   first_clq_next = find(clique_ix_to_frame_ix == frame_ix+1, 1,'First');

   %% Adjacency between prev and next, from adjacency betweeen superpsts:
   % i_left and i_right will be used for the match, but first we need to
   % re-tessellate them!!!
   
   % prev_i_L_superpst
   % next_i_L_superpst  
   %%
   i_left        = prev_i_L_superpst;
   i_left        = thin_and_recolor_image(i_left);
   i_left_bdries = i_left == 0;    
   %%   
   i_right        = next_i_L_superpst;
   i_right        = thin_and_recolor_image(i_right);
   i_right_bdries = i_right == 0;
   %%
   lbls_i_left     = unique(i_left(:));
   lbls_i_right    = unique(i_right(:));
   lbls_i_left(lbls_i_left == 0)   = [];
   lbls_i_right(lbls_i_right == 0) = [];

   n_segs_left     = numel(unique(i_left(:)))-1;
   n_segs_right    = numel(unique(i_right(:)))-1;
   % Now, map i_left & i_right to:
   % 1....N

   % Map i_left
   keys    = lbls_i_left;
   values  = 1:n_segs_left;
   norm_left_mapping(keys) = values;

   i_left(i_left_bdries) = 1;
   i_left = norm_left_mapping(i_left);
   i_left(i_left_bdries) = 0;

   % Map i_right
   keys    = lbls_i_right;
   values  = 1:n_segs_right;
   norm_right_mapping(keys) = values;

   i_right(i_right_bdries) = 1;
   i_right = norm_right_mapping(i_right);

   % Shift i_right by n_segs_prev
   i_right = i_right + n_segs_left;    
   i_right(i_right_bdries) = 0;    

   % Compute adjacency with imRAG_only_3D
   % Get everything ready for imrag
   i_L_adj = zeros(nR, nC, 3);

   % Prior to getting the adjacency, we get a topologically equivalent,
   % 1pix thin tessellation...   
   
   i_L_adj(:,:,1) =  i_left;
   i_L_adj(:,:,3) =  i_right;
   n_segs_total_match  = numel(unique(i_L_adj)) -1;

   % We need signed integers at least for the arguments of imRAG
   % but sparse actually needs double...
   if strcmp(adj_type, 'norm_disp')
      rag_i_lbls       = double(imRAG_only_3D(int32(i_L_adj)));
   else
      rag_i_lbls       = double(imRAG_only_3D_large_disp(int32(i_L_adj)));
   end 
 
   %%   
   % Undo the shifting for the sparse matrices:
   rag_i_lbls(:,1)  = rag_i_lbls(:,1); % No un-shifting required 
   rag_i_lbls(:,2)  = rag_i_lbls(:,2) - n_segs_left;

   n_sppst_edges     = size(rag_i_lbls,1);
   sppst_edges       = ones(n_sppst_edges, 1);

   sp_adj_spsst      = sparse(rag_i_lbls(:,1), rag_i_lbls(:,2), sppst_edges, ...
                     double(n_segs_left),double(n_segs_right));

   % Make sure the multiplication later makes sense!
   if isempty(next_spt_M) 
      next_spt_M = (sp_adj_spsst'); 
      next_spt_M = (next_spt_M(:,[]))'; % Notice the transponse here
   end                  
   
   if isempty(prev_spt_M) 
      prev_spt_M = sp_adj_spsst'; 
      prev_spt_M = prev_spt_M([],:);    % This one is not transposed
      
   end
   prev_next_adj     = (prev_spt_M * sp_adj_spsst * next_spt_M');

   [rows{frame_ix},cols{frame_ix},vals{frame_ix}]  = find(prev_next_adj);    
    
    
   rows{frame_ix}    = rows{frame_ix} + first_clq_prev-1;
   cols{frame_ix}    = cols{frame_ix} + first_clq_next-1;   

end
% rows
% size(rows)
% whosb rows
rows = cellfun(@(x) x(:), rows, 'UniformOutput', false);
cols = cellfun(@(x) x(:), cols, 'UniformOutput', false);

rows = cell2mat(rows);
cols = cell2mat(cols);
vals = ones(size(rows,1),1);
clq_3D_adj = sparse(rows, cols, vals, ...
      double(n_total_cliques),double(n_total_cliques));
   
disp(['All done in ' secs2hms(toc(tic_all_frames))]);
n_links               = nnz(clq_3D_adj);
disp(['Total # of links: ' num2str(n_links)]);


%% We now compute the link IDs

% sp_rows           = zeros(n_links,1);
% sp_cols           = zeros(n_links,1);
% sp_link_ids_vals  = zeros(n_links,1);
% 
% link_ix = 0;
% for this_clique_ix = 1:n_total_cliques
%   nghbors_cliques_ix = find(clq_3D_adj(this_clique_ix,:));
%   nghbors_cliques_ix = nghbors_cliques_ix(:)';
%   for nghbor_clique_ix    = nghbors_cliques_ix
%      link_ix = link_ix + 1;
% 
%      sp_rows(link_ix) = this_clique_ix;
%      sp_cols(link_ix) = nghbor_clique_ix;
%      sp_link_ids_vals(link_ix) = link_ix;
%   end
% end
% links_As = sp_rows;
% links_Bs = sp_cols;
% 
% clq_3D_link_ids             = sparse (sp_rows, sp_cols, sp_link_ids_vals, n_total_cliques,n_total_cliques);


disp(' ');
msg_str = 'Determining link IDs from adjacency';
t_measure = tic;
disp([ msg_str '...']);

A =  clq_3D_adj;
B = A;              %# Initialize B
C = A;              %# Initialize C
mask = logical(A);  %# Create a logical mask using A

[r,c] = find(A);    %# Find the row and column indices of non-zero values
index = c + (r - 1).*size(A,2);  % Compute the row-first linear index
[~,order] = sort(index);         % Compute the row-first order with
[~,order] = sort(order);         % two sorts
B(mask) = index;
C(mask) = order;


[r c v] = find(C); 
[~, idx] = sort(v); 
r = r(idx); c = c(idx);
links_As = r;
links_Bs = c;
clq_3D_link_ids = C;
disp(['done in ' secs2hms(toc(t_measure)) ' for "' msg_str '"']);   
%%



% Get links fwd and bckwd
all_fwd_links_per_clique   = cell(n_total_cliques,1);
all_bckwd_links_per_clique = cell(n_total_cliques,1);

tic
disp('Computing forward and backward links');

link_indices = [1:n_links]';
subs = link_indices;
vals = links_Bs;
all_bckwd_links_per_clique_from_accumarray = accumarray(vals, link_indices, [], @(x) {x});
vals = links_As;
all_fwd_links_per_clique_from_accumarray   = accumarray(vals, link_indices, [], @(x) {x});

all_bckwd_links_per_clique(1:numel(all_bckwd_links_per_clique_from_accumarray)) = ...
   all_bckwd_links_per_clique_from_accumarray;
all_fwd_links_per_clique(1:numel(all_fwd_links_per_clique_from_accumarray))     = ...
   all_fwd_links_per_clique_from_accumarray;

% OLDER:
% for clique_ix = 1:n_total_cliques
%    all_bckwd_links_per_clique{clique_ix}  = find(logical(ismembc2(link_Bs, cast(clique_ix, class(link_Bs)))));
%    all_fwd_links_per_clique{clique_ix}    = find(logical(ismembc2(link_As, cast(clique_ix, class(link_As)))));
% end
disp(['Done in ' secs2hms(toc)]);


link_rep_data              = var2struct(  clq_3D_adj, ...
                                          clq_3D_link_ids, ...
                                          links_As, ...
                                          links_Bs, ...                                          
                                          n_links, ...
                                          all_bckwd_links_per_clique, ...
                                          all_fwd_links_per_clique, ...
                                          'no');
catch me
   me.getReport()
   if ispc
      keyboard;
   end
end
end