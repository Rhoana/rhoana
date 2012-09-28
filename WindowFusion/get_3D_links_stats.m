function link_stats   = get_3D_links_stats(...
                                       clique_rep_data, ...
                                       link_rep_data, ...
                                       i_s_bw_non_cent, ...
                                       i_s_bw_non_cent_displaced)

% DISPLACEMENT AND OVERLAP (DICE)
  struct2var(clique_rep_data.spt_maps);
  struct2var(link_rep_data);
    
  %% XY displacement:
  disp(' ');
  disp('Determining link IDs from adjacency..');
  
  % This could in principle be done with the big (non-centered) logical maps
  % This has to be done with the non-displaced ones

%   sp_rows = zeros(n_links,1);
%   sp_cols = zeros(n_links,1);
%   sp_link_ids_vals = zeros(n_links,1);
%   link_ix = 0;

%   % We have the 3D adjacency matrix!
%   for clique_from_ix = 1:n_total_cliques
%      nghbors_cliques_ix = find(clq_3D_adj(clique_from_ix,:));
%      nghbors_cliques_ix = nghbors_cliques_ix(:)';
%      for nghbor_clique_ix    = nghbors_cliques_ix
%         link_ix = link_ix + 1;
%         % Measure symmetric difference:
%         sp_rows(link_ix) = clique_from_ix;
%         sp_cols(link_ix) = nghbor_clique_ix;
%         sp_link_ids_vals(link_ix) = link_ix;
%      end
%   end
%   
%   clq_3D_link_ids = sparse (sp_rows, sp_cols, sp_link_ids_vals, n_total_cliques,n_total_cliques);

% Get the link IDs
tic_all_frames = tic;   
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
clq_3D_link_ids = C;

disp(['Done in ' secs2hms(toc(tic_all_frames))]);

  %%
   disp(' ');
   disp('Determining frame-to-frame ovlap in centered bitmaps (FAST method) ...');
   tic_all_frames = tic;
   % We have the 3D adjacency matrix!
   link_ix = 0;

   links_fwd_overlap    = zeros(n_links, 1);
   links_bckwd_overlap  = zeros(n_links, 1);

   for  clique_from_ix = 1:n_total_cliques
      nghbors_clique_indices = find(clq_3D_adj(clique_from_ix,:));

      % Get i_bw_this_clique
      i_bw_from_clique_non_cent           = squeeze(i_s_bw_non_cent(:,:,clique_from_ix));
      i_bw_from_clique_non_cent_disp      = squeeze(i_s_bw_non_cent_displaced(:,:,clique_from_ix));      
      
      n_clique_from_non_cent              = sum(sum(i_bw_from_clique_non_cent));
      n_clique_from_non_cent_disp         = sum(sum(i_bw_from_clique_non_cent_disp));

      nghbors_clique_indices = nghbors_clique_indices(:)';

     for nghbor_clique_ix    = nghbors_clique_indices
         link_ix = link_ix + 1;
         link_id = clq_3D_link_ids(clique_from_ix,nghbor_clique_ix);
         
         % This tells us that linkIDs are row-order first!
         if link_ix ~= link_id; disp('Differentl link ids!!');  end
         
         if ~mod(link_ix,5000) disp(link_ix); end
         
         i_bw_to_clique_non_cent           = squeeze(i_s_bw_non_cent(:,:,nghbor_clique_ix));
         i_bw_to_clique_non_cent_disp      = squeeze(i_s_bw_non_cent_displaced(:,:,nghbor_clique_ix));                    
         
         n_clique_to_non_cent              = sum(sum(i_bw_to_clique_non_cent));
         n_clique_to_non_cent_disp         = sum(sum(i_bw_to_clique_non_cent_disp));

         % Forward intersection:
         n_int_fwd                         = sum(sum(i_bw_from_clique_non_cent_disp & i_bw_to_clique_non_cent));

         % Backward intersection:
         n_int_bckwd                       = sum(sum(i_bw_to_clique_non_cent_disp & i_bw_from_clique_non_cent));      
         
        % Measure symmetric difference forward and backward
         links_fwd_overlap(link_id)        = n_int_fwd/n_clique_from_non_cent_disp;
         links_bckwd_overlap(link_id)      = n_int_bckwd/n_clique_to_non_cent_disp;
         true;
     end
   end
   
   disp(['Done in ' secs2hms(toc(tic_all_frames))]);
   
   %  disp('We take care of NaNs in cent_dice because of the downsampling');
   %  links_cent_dice(isnan(links_cent_dice)) = 0.5;
    

   %% 
link_stats = var2struct(...
   links_fwd_overlap, ...
   links_bckwd_overlap, ...
   'no');

end