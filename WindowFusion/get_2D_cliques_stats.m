function clique_stats = get_2D_cliques_stats(clique_rep_data,i_stacks, uv_flows, dwnsmpling_factor)

   % SIZES AND CENTROIDS  
   % This is what we need:
struct2var(clique_rep_data.spt_maps);
spt_mf_M = clique_rep_data.spt_mf_M;


i_s_L_sppst = i_stacks.i_s_L_sppst;

clique_to_xC       = zeros(n_total_cliques,1);
clique_to_yC       = zeros(n_total_cliques,1);
% clique_x_flows     = zeros(n_total_cliques,1);
% clique_y_flows     = zeros(n_total_cliques,1);
% clique_flow_mags   = zeros(n_total_cliques,1);

clique_sizes    = zeros(n_total_cliques,1);

disp(['Getting 2D stats for ' num2str(n_total_cliques) ' cliques in all frames but the last one ...']);
n_frames_to_process = numel(n_cliques_per_frame);

cliques_on_all_frames_but_last_one = find(clique_ix_to_frame_ix ~=  n_frames_to_process);
cliques_on_all_frames_but_last_one = cliques_on_all_frames_but_last_one(:)';
%%
% # amvr TO DO:
% We can see how much the forward and the backward flow agree when
% measuring this.

t_all_cliques = tic;
for frame_ix = 1:n_frames_to_process -1 
   all_cliques_this_frame = find(clique_ix_to_frame_ix ==  frame_ix);
   all_cliques_this_frame = all_cliques_this_frame(:)';
%    uv_x = sq(uv_flows.fwd_x(:,:, frame_ix));
%    uv_y = sq(uv_flows.fwd_y(:,:, frame_ix));      

   for clique_ix = all_cliques_this_frame
      i_L_sppst               = squeeze(i_s_L_sppst(:,:,frame_ix));
      sppst_ixs_this_clique   = [find(spt_mf_M(clique_ix,:))]';   
      sppst_lbls_this_clique  = sppst_ix_to_sppst_lbl_ix(sppst_ixs_this_clique);   

       i_bw_2D  = ...
            logical(ismembc2(i_L_sppst, ...
            cast(sort(sppst_lbls_this_clique), class(i_L_sppst))));           

      [yi xi]                   = find(i_bw_2D);
      if ~isempty(xi) % To avoid NaN values in clique_stats
         clique_to_xC(clique_ix)   = mean(xi);
         clique_to_yC(clique_ix)   = mean(yi);
         % Get the size of each 2D segment
          clique_sizes(clique_ix) = sum(sum(i_bw_2D));         

%          clique_flow_mags(clique_ix)   = median(sqrt(uv_x(i_bw_2D).^2 + uv_y(i_bw_2D).^2));
%          clique_x_flows(clique_ix)     = median(uv_x(i_bw_2D));
%          clique_y_flows(clique_ix)     = median(uv_y(i_bw_2D));
      end
   end
end
disp(['Done in ' secs2hms(toc(t_all_cliques))]);

%% Compute the rest of the stats for the last frame:
disp(['Getting 2D stats for ' num2str(n_total_cliques) ' cliques on the last frame...']);
frame_ix = n_frames_to_process;
all_cliques_this_frame = find(clique_ix_to_frame_ix ==  frame_ix);
all_cliques_this_frame = all_cliques_this_frame(:)';
t_all_cliques = tic;
for clique_ix = all_cliques_this_frame
   i_L_sppst               = squeeze(i_s_L_sppst(:,:,frame_ix));
   sppst_ixs_this_clique   = [find(spt_mf_M(clique_ix,:))]';   
   sppst_lbls_this_clique  = sppst_ix_to_sppst_lbl_ix(sppst_ixs_this_clique);   

    i_bw_2D  = ...
         logical(ismembc2(i_L_sppst, ...
         cast(sort(sppst_lbls_this_clique), class(i_L_sppst))));           

   [yi xi]                   = find(i_bw_2D);
   if ~isempty(xi) % To avoid NaN values in clique_stats
      clique_to_xC(clique_ix)   = mean(xi);
      clique_to_yC(clique_ix)   = mean(yi);
      % Get the size of each 2D segment
       clique_sizes(clique_ix) = sum(sum(i_bw_2D));         
   end
end
disp(['Done in ' secs2hms(toc(t_all_cliques))]);
%%
% We can probably do this with copy_struct and var2struct
% clique_stats.clique_x_flows    = clique_x_flows;
% clique_stats.clique_y_flows    = clique_y_flows;
% clique_stats.clique_flow_mags  = clique_flow_mags;
clique_stats.clique_sizes   = clique_sizes;
clique_stats.clique_to_xC   = clique_to_xC;
clique_stats.clique_to_yC   = clique_to_yC;
end
