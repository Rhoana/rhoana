function [spt_mf_M, spt_maps]  = spt_remove_repetitions_and_empty_cliques(spt_mf_M, spt_maps)
   n_frames_to_process = numel(spt_maps.n_cliques_per_frame);
   spt_Ms_per_frame = cell(n_frames_to_process,1);

   for frame_ix = 1:n_frames_to_process
      spt_M_this_frame            = spt_mf_M( spt_maps.clique_ix_to_frame_ix == frame_ix, ...
                                              spt_maps.sppst_ix_to_frame_ix ==  frame_ix);

      %% Repetitions:
      which_rows_to_consider = true(size(spt_M_this_frame,1),1);
      which_indices_to_keep_from_repetitions    = find_unique_rows(spt_M_this_frame, which_rows_to_consider);
     
      %% Empty cliques:      
      nR= size(spt_M_this_frame,1);

      sppsts = cell(nR, 1);
      for row_ix = 1:nR
         sppsts{row_ix} = find(spt_M_this_frame(row_ix,:));
      end

      which_row_is_emmpty = cellfun(@(x) isempty(x), sppsts, 'UniformOutput', true);
      indices_to_keep_from_empty_check = ~which_row_is_emmpty;            
      
      which_rows_to_keep  = which_indices_to_keep_from_repetitions & indices_to_keep_from_empty_check;
      spt_Ms_per_frame{frame_ix} = spt_M_this_frame(which_rows_to_keep,:);   
   end

   [spt_mf_M, spt_maps] = build_spt_mf_M_and_maps(spt_Ms_per_frame);

end