function [spt_M,i_s_L_sppst, cliques_per_solution] = ... 
   build_spt_M_cube_no_clique_props(i_s_L_solutions, i_s_L_sppst)

[nR, nC, n_frames]   = size(i_s_L_solutions{1});
n_sols               = numel(i_s_L_solutions);

n_labels_so_far = 0;
for frame_ix = 1:n_frames
   i_zero_this_frame       = squeeze(i_s_L_sppst(:,:,frame_ix) == 0);
   i_L_sppst               = uint32(labelmatrix(bwconncomp(squeeze(i_s_L_sppst(:,:,frame_ix)),4)));
   i_L_sppst               = i_L_sppst + n_labels_so_far;
   n_labels_so_far         = n_labels_so_far + numel(faster_unique(i_L_sppst(:)))-1;
   i_L_sppst(i_zero_this_frame) = 0;

   i_s_L_sppst(:,:,frame_ix) = i_L_sppst;
end
max_sppst_label = max(i_s_L_sppst(:));
%%
sol_lbls_to_sppst_lbls = cell(n_sols, 1);
for sol_ix = 1:n_sols
   i_s_L_solution = i_s_L_solutions{sol_ix};

   lbls_i_s_L_solution = faster_unique(i_s_L_solution(:));
   lbls_i_s_L_solution(lbls_i_s_L_solution == 0) = [];
   sol_lbls_to_sppst_lbls{sol_ix} = arrayfun(@(x) faster_unique(i_s_L_sppst(i_s_L_solution == x)), lbls_i_s_L_solution, 'UniformOutput', false);   
   sol_lbls_to_sppst_lbls{sol_ix} = cellfun(@(x) x(x ~= 0),sol_lbls_to_sppst_lbls{sol_ix}, 'UniformOutput', false);
end

%%
n_cliques_per_solution = cellfun(@(x) numel(x), sol_lbls_to_sppst_lbls, 'UniformOutput', true);
n_total_cliques        = sum(n_cliques_per_solution);


%%
sppst_lbls              = cat(1,sol_lbls_to_sppst_lbls{:});

%%
lin_cols = sppst_lbls;
lin_rows = num2cell((1:numel(lin_cols))');
%%
lin_rows = cell2mat(cellfun(@(x,y) repmat(x, numel(y),1), lin_rows, lin_cols, 'UniformOutput', false));
lin_cols = cell2mat(lin_cols);
n_nnz    = numel(lin_cols);
lin_vals = ones(n_nnz, 1);
%%
nR = max(lin_rows);
nC = max_sppst_label;


if isempty(nR)
   spt_M = [];
   return;
   
else   
   try
      spt_M    = sparse(double(lin_rows), double(lin_cols), lin_vals,...
                        double(nR),double(nC),n_nnz);
   catch me
      if ispc,
         keyboard;
      end
   end


   which_are_unique = find_unique_rows(spt_M, true(size(spt_M,1),1));

   spt_M                = spt_M(which_are_unique, :);

end


%% Identify the cliques from each solution on the matrix!
%% We find the cliques belonging to each solution.
% Check out find_my_cliques_from_i_s_tessellation.m too!
cliques_per_solution = cell(n_sols,1);

[~, nC] = size(spt_M);
disp('Identifying cliques per solution');
t_cliques_per_sol = tic;
for sol_ix = 1:n_sols

   sppst_lbls_this_sol  = sol_lbls_to_sppst_lbls{sol_ix};
   n_cliques_this_sol   = numel(sppst_lbls_this_sol);

   clique_indices_for_this_sol = zeros(n_cliques_this_sol,1);
   for clique_ix = 1:n_cliques_this_sol
      n_sppst_ixs_this_clique = numel(sppst_lbls_this_sol{clique_ix});
      linear_vector =  -1 *ones(nC, 1);
      linear_vector(sppst_lbls_this_sol{clique_ix}) = 1;
      row_candidates = spt_M * linear_vector;
      if ~isempty(find(row_candidates == n_sppst_ixs_this_clique, 1, 'first'))
         clique_indices_for_this_sol(clique_ix) = find(row_candidates == n_sppst_ixs_this_clique, 1, 'first');   
      else
         clique_indices_for_this_sol(clique_ix) = 0;
      end
   end
   cliques_per_solution{sol_ix} = clique_indices_for_this_sol;
end
disp(['Done with identifying cliques per solution in ' secs2hms(toc(t_cliques_per_sol))]);
end