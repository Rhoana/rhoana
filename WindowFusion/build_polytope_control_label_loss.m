function [lhs_polytope, ...
         A_polytope, ...
         rhs_polytope] = build_polytope_control_label_loss(...
   vars_cliques, ....
   vars_links, ...
   n_variables, ...
   link_As, ...
   link_Bs, ...
   all_bckwd_links_per_clique,...
   all_fwd_links_per_clique, ...   
   n_cliques, ...
   n_frames_to_process, ...
   clique_2D_sizes, ...
   clique_ix_to_frame_ix, ...
   clq_cost_cntrnts) 
try
%% Some local definitions:
struct2var(clq_cost_cntrnts);
% loss_type
% loss_limits

disp(['We only tolerate losing this # of pixels: ' num2str(loss_limits.absolute_pixel)])

%% ------------------------------------


%% ===============================================
% Set of inequalities:
%%===============================================
    
%% Collect the indices in cell arrays. This is faster than indexing directly
% into the sparse matrix!:
  % Two equations per 2D clique:
A_cols   = cell(2*n_cliques, 1);
A_values = cell(2*n_cliques, 1);
rhs_values    = zeros(2*n_cliques, 1);

ineq_ix = 0;
for clique_2D_ix = 1:n_cliques
   this_frame_ix = clique_ix_to_frame_ix(clique_2D_ix);


   which_links_bckwd  = all_bckwd_links_per_clique{clique_2D_ix};
   which_links_fwd    = all_fwd_links_per_clique{clique_2D_ix};
  
  if  this_frame_ix >1
     %% ---------------
     %% Backward links  
     %% ---------------
     if isempty(which_links_bckwd)
        continue;
     end
     
     % Segment:
     this_clique_vars         = reshape(vars_cliques(clique_2D_ix), [],1);
     ineq_cols_1              = this_clique_vars;
     vals_1                   = ones(numel(this_clique_vars), 1);

     % Links
     this_clique_links_vars  = reshape(vars_links(which_links_bckwd), [],1);
     ineq_cols_2              = this_clique_links_vars;
     vals_2                   = ones(numel(this_clique_links_vars), 1);

     % Get the areas:
     this_clique_links_to_cliques = link_As(which_links_bckwd);
     % this_clique_links_to_vars    =  vars_cliques(this_clique_links_to_cliques);
     this_clique_links_to_areas   = clique_2D_sizes(this_clique_links_to_cliques);
     this_clique_area             = clique_2D_sizes(clique_2D_ix);  

     %%  this_clique_area <= SUM(this_clique_links_to_areas)
     % Segment:
%      vals_1 = this_clique_area * vals_1 - loose_margin*this_clique_area;
      switch loss_type.which 
         case loss_type.percentage
            vals_1 = this_clique_area * vals_1 - loss_limits.percentage_limit * this_clique_area;
         case loss_type.absolute_pixel
            vals_1 = this_clique_area * vals_1 - loss_limits.absolute_pixel;
      end
         
     % Links:
     vals_2 = (-1) * this_clique_links_to_areas .* vals_2; 

     %% Equation:
     ineq_ix = ineq_ix + 1;
     A_cols{ineq_ix}    = cat(1, ineq_cols_1, ineq_cols_2);
     A_values{ineq_ix}  = cat(1, vals_1, vals_2);  
     rhs_values(ineq_ix)     = 0;
  end
  if this_frame_ix < n_frames_to_process
     %% ---------------  
     %%   Forward links:
     %% ---------------
     if isempty(which_links_fwd)
        continue;
     end
     % Segment
     this_clique_vars         = reshape(vars_cliques(clique_2D_ix), [],1);
     ineq_cols_1              = this_clique_vars;
     vals_1                   = ones(numel(this_clique_vars), 1);

     % Links
     this_clique_vars         = reshape(vars_links(which_links_fwd), [],1);
     ineq_cols_2              = this_clique_vars;
     vals_2                   = ones(numel(this_clique_vars), 1);  

     % Get the areas:
     this_clique_links_to_cliques = link_Bs(which_links_fwd);
     % this_clique_links_to_vars    =  vars_cliques(this_clique_links_to_cliques);
     this_clique_links_to_areas   = clique_2D_sizes(this_clique_links_to_cliques);
     this_clique_area             = clique_2D_sizes(clique_2D_ix);  

     %%  this_clique_area <= SUM(this_clique_links_to_areas)
     % Segment:
      switch loss_type.which 
         case loss_type.percentage
            vals_1 = this_clique_area * vals_1 - loss_limits.percentage_limit * this_clique_area;
         case loss_type.absolute_pixel
            vals_1 = this_clique_area * vals_1 - loss_limits.absolute_pixel;
      end

     % Links:
     vals_2 = (-1) * this_clique_links_to_areas .* vals_2; 

     %% Equation:
     ineq_ix = ineq_ix + 1;
     A_cols{ineq_ix}    = cat(1, ineq_cols_1, ineq_cols_2);
     A_values{ineq_ix}  = cat(1, vals_1, vals_2);  
     rhs_values(ineq_ix)     = 0;
  end
  
end
A_cols(ineq_ix+1:end)      = [];
A_values(ineq_ix+1:end)    = [];
rhs_values(ineq_ix+1:end)  = [];
% Build back the indices

A_cols = cellfun(@(x) double(x), A_cols, 'UniformOutput', false);

lin_rows = num2cell((1:numel(A_cols))');
lin_rows = cell2mat(cellfun(@(x,y) repmat(x, numel(y),1), lin_rows, A_cols, 'UniformOutput', false));

lin_cols      = cell2mat(A_cols);
lin_values    = cell2mat(A_values);

nRows = max(lin_rows);
nCols = n_variables;
nzmax = numel(lin_rows);

% lin_rows
% lin_cols
% lin_values
A_polytope    = sparse(double(lin_rows), double(lin_cols), lin_values,...
   double(nRows),double(nCols),nzmax);
rhs_polytope  = sparse(rhs_values);
lhs_polytope  = repmat(-Inf, numel(rhs_polytope), 1);

catch me
   if ispc, warn_me, keyboard; end
end