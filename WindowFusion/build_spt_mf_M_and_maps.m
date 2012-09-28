function [spt_mf_M, spt_maps] = build_spt_mf_M_and_maps(spt_Ms_per_frame)

   n_frames_to_process = numel(spt_Ms_per_frame);
   spt_mf_M = blkdiag(spt_Ms_per_frame{:});

   % Get nRs and nCs
   clq_start_frame_indices    = zeros(n_frames_to_process, 1);
   clq_end_frame_indices      = zeros(n_frames_to_process, 1);

   sppst_start_frame_indices  = zeros(n_frames_to_process, 1);
   sppst_end_frame_indices    = zeros(n_frames_to_process, 1);

   nRs = zeros(n_frames_to_process, 1);
   nCs = zeros(n_frames_to_process, 1);
   for frame_ix=1:n_frames_to_process
      [nRs(frame_ix) nCs(frame_ix)] = size(spt_Ms_per_frame{frame_ix});

      clq_start_frame_indices(frame_ix)   = sum(nRs(1:frame_ix-1))+1;
      clq_end_frame_indices(frame_ix)     = clq_start_frame_indices(frame_ix) + nRs(frame_ix)-1;

      sppst_start_frame_indices(frame_ix) = sum(nCs(1:frame_ix-1))+1;
      sppst_end_frame_indices(frame_ix)   = sppst_start_frame_indices(frame_ix) + nCs(frame_ix)-1;
   end

   %% Build spt_maps:

   % clique_ix_to_frame_ix
   cliques_per_frame          = num2cell(nRs);
   frame_ix_for_each_submat   = num2cell([1:n_frames_to_process]');
   clique_ix_to_frame_ix      = cellfun(@(x,y) y*ones(x,1), cliques_per_frame, frame_ix_for_each_submat, 'UniformOutput', false);
   clique_ix_to_frame_ix      = cell2mat(clique_ix_to_frame_ix);
   clique_ix_to_frame_ix      = uint32(clique_ix_to_frame_ix);

   % sppst_ix_to_frame_ix
   cliques_per_frame          = num2cell(nCs);
   frame_ix_for_each_submat   = num2cell([1:n_frames_to_process]');
   sppst_ix_to_frame_ix       = cellfun(@(x,y) y*ones(x,1), cliques_per_frame, frame_ix_for_each_submat, 'UniformOutput', false);
   sppst_ix_to_frame_ix       = cell2mat(sppst_ix_to_frame_ix);
   sppst_ix_to_frame_ix       = uint32(sppst_ix_to_frame_ix);

   % sppst_ix_to_sppst_lbl_ix
   n_sppst_labels_per_array   = num2cell(nCs);
   sppst_ix_to_sppst_lbl_ix   = cellfun(@(x) [1:x]', n_sppst_labels_per_array, 'UniformOutput', false);
   sppst_ix_to_sppst_lbl_ix   = cell2mat(sppst_ix_to_sppst_lbl_ix);
   sppst_ix_to_sppst_lbl_ix   = uint32(sppst_ix_to_sppst_lbl_ix);

   % sppst_lbl_to_sppst_ix
   sppst_lbl_to_sppst_ix      = mat2cell([sppst_start_frame_indices sppst_end_frame_indices],ones(n_frames_to_process,1),2);
   sppst_lbl_to_sppst_ix      = cellfun(@(x) [x(1):x(2)]', sppst_lbl_to_sppst_ix, 'UniformOutput', false);
   
   % n_2D_cliques per frame:
   n_cliques_per_frame = zeros(n_frames_to_process,1);
   for frame_ix = 1:n_frames_to_process
      n_cliques_per_frame(frame_ix)             = size(spt_mf_M(clique_ix_to_frame_ix == frame_ix, ...
                                                                   sppst_ix_to_frame_ix ==  frame_ix),1);
   end   
   %%

n_total_cliques = sum(n_cliques_per_frame);

spt_maps = var2struct(...
   clique_ix_to_frame_ix, ...
   sppst_ix_to_frame_ix, ...
   sppst_ix_to_sppst_lbl_ix, ...
   sppst_lbl_to_sppst_ix, ...
   n_cliques_per_frame, ...
   n_total_cliques, ...
   'no');
   
   
end
