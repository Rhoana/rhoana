function [i_s_bw_cent, i_s_bw_non_cent, smll_bw_mtdata] = get_i_s_bw_all_2D_cliques(clq_2D_data,i_stacks, target_comp_bw_size)

spt_mf_M     = clq_2D_data.spt_mf_M;
spt_maps     = clq_2D_data.spt_maps;

n_2D_cliques               = sum(spt_maps.n_cliques_per_frame);
sppst_ix_to_sppst_lbl_ix   = spt_maps.sppst_ix_to_sppst_lbl_ix;
clique_ix_to_frame_ix      = spt_maps.clique_ix_to_frame_ix;

i_s_L_sppst = i_stacks.i_s_L_sppst;
  
  
%% Stack of binary bitmaps:
disp(' ');
disp('Building image stack with binary bitmaps..');
disp(['Total # of 2D cliques: ' num2str(n_2D_cliques)]);

[nR nC nF] = size(i_s_L_sppst);

[~, which_is_larger]         = max([nR, nC]);

if which_is_larger == 1
   i_tmp             = imresize(false(nR, nC), [target_comp_bw_size, NaN], 'nearest');
   dwnsmpling_factor = nR/target_comp_bw_size;
else
   i_tmp             = imresize(false(nR, nC), [NaN, target_comp_bw_size], 'nearest');
   dwnsmpling_factor = nC/target_comp_bw_size;
end

[smll_nR, smll_nC]   = size(i_tmp);

smll_bw_mtdata.n_frames          = nF;
smll_bw_mtdata.height            = smll_nR;
smll_bw_mtdata.width             = smll_nC;
smll_bw_mtdata.dwnsmpling_factor = dwnsmpling_factor;

% Center pixel of the image:
I_Yc                = round(smll_nR/2);
I_Xc                = round(smll_nC/2);

i_s_bw_cent       = false(uint32([smll_nR, smll_nC, n_2D_cliques]));
i_s_bw_non_cent   = false(uint32([smll_nR, smll_nC, n_2D_cliques]));

t_building_i_bw_stack = tic;
for  this_clique_ix = 1:n_2D_cliques     

   % Get i_bw_this_clique
   frame_ix    = clique_ix_to_frame_ix(this_clique_ix);
   i_L_sppst = squeeze(i_s_L_sppst(:,:,frame_ix));
   sppst_ixs_this_clique   = [find(spt_mf_M(this_clique_ix,:))]';   
   sppst_lbls_this_clique  = sppst_ix_to_sppst_lbl_ix(sppst_ixs_this_clique);   

    i_bw_this_clique  = ...
         logical(ismembc2(i_L_sppst, ...
         cast(sort(sppst_lbls_this_clique), class(i_L_sppst))));     
   try
   i_bw_this_clique = imresize(i_bw_this_clique, [smll_nR smll_nC], 'nearest');
   %% Fast, re-centering      
   bw = i_bw_this_clique;
   
   if any(bw(:))
      [yi xi]         = find(bw);
      clique_xC  = mean(xi);
      clique_yC  = mean(yi);

      numDimsA = ndims(bw);
      sizeA    = size(bw);

      p(1) = round(I_Yc-clique_yC);
      p(2) = round(I_Xc-clique_xC);

      idx = cell(1, numDimsA);

      % Loop through each dimension of the input matrix to calculate shifted indices
      for k = 1:numDimsA
          m      = sizeA(k);
          idx{k} = mod((0:m-1)-p(k), m)+1;
      end
   
       i_s_bw_cent(:,:,this_clique_ix)     = i_bw_this_clique(idx{:});
       i_s_bw_non_cent(:,:,this_clique_ix) = i_bw_this_clique;      
   else
       i_s_bw_cent(:,:,this_clique_ix)     = bw;
       i_s_bw_non_cent(:,:,this_clique_ix) = bw;      
   end
   
   catch me
      warn_me
      keyboard
   end
end
disp(['Done in ' secs2hms(toc(t_building_i_bw_stack))]);

end
