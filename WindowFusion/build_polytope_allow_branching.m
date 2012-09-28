function [lhs_polytope, ...
         A_polytope, ...
         rhs_polytope]= build_polytope_allow_branching(LP_mtdata)
%    vars_cliques, ....
%    vars_links, ...
%    n_total_variables, ...
%    link_As, ...
%    link_Bs, ...
%    all_bckwd_links_per_clique,...
%    all_fwd_links_per_clique, ...      
%    n_total_cliques, ...
%    clq_cost_cntrnts)


struct2var(LP_mtdata)
% This has to be fixed!! At the moment it forces links to be activated 
% if a segment is activated...

%% Collect the indices in cell arrays. This is faster than indexing directly
% into the sparse matrix!:
  % Two equations per 2D clique:
A_cols   = cell(4*n_total_cliques, 1);
A_values = cell(4*n_total_cliques, 1);
rhs_values    = zeros(4*n_total_cliques, 1);

n_max_branches = clq_cost_cntrnts.n_max_branches;

ineq_ix = 0;
for clique_ix = 1:n_total_cliques

%    fwd               =  find(clq_3D_cent(clique_ix,:));   
%    bckwd             =  find(clq_3D_cent(:,clique_ix));

   which_links_bckwd  = all_bckwd_links_per_clique{clique_ix};
   which_links_fwd    = all_fwd_links_per_clique{clique_ix};
  
  %% ---------------
  %% Backward links  
  %% ---------------
  ineq_ix = ineq_ix + 1;
  % Segment:
  this_clique_vars         = reshape(vars_cliques(clique_ix), [],1);
  ineq_cols_1              = this_clique_vars;
  vals_1                   = ones(numel(this_clique_vars), 1);
  
  % Links
  this_clique_vars         = reshape(vars_links(which_links_bckwd), [],1);
  ineq_cols_2              = this_clique_vars;
  vals_2                   = ones(numel(this_clique_vars), 1);
  
  %% SUM(links) - M segment <= 0
  % Segment:
  M = numel(vals_2);
  M = min(n_max_branches,M);
  vals_1 = (-M)  * vals_1;
  
  % Links:
  vals_2 = 1 * vals_2; 
  
  %% Equation:
  A_cols{ineq_ix}    = cat(1, ineq_cols_1, ineq_cols_2);
  A_values{ineq_ix}  = cat(1, vals_1, vals_2);  
  rhs_values(ineq_ix)     = 0;

   % I don't think we need this equation: Segment <= SUM(links) (diff <= 0)

  
  %% ---------------  
  %%   Forward links:
  %% ---------------
  ineq_ix = ineq_ix + 1; 
  
  % Segment
  this_clique_vars         = reshape(vars_cliques(clique_ix), [],1);
  ineq_cols_1              = this_clique_vars;
  vals_1                   = ones(numel(this_clique_vars), 1);

  % Links
  this_clique_vars         = reshape(vars_links(which_links_fwd), [],1);
  ineq_cols_2              = this_clique_vars;
  vals_2                   = ones(numel(this_clique_vars), 1);  
  
  %% SUM(links) - M segment <= 0
  % Segment:
  M = numel(vals_2);
  M = min(n_max_branches,M);
  vals_1 = (-M)  * vals_1;
  
  % Links:
  vals_2 = 1 * vals_2; 
  
  %% Equation:
  A_cols{ineq_ix}    = cat(1, ineq_cols_1, ineq_cols_2);
  A_values{ineq_ix}  = cat(1, vals_1, vals_2);  
  rhs_values(ineq_ix)     = 0;
  
 
 
  

%   %% Segment <= SUM(links) (diff <= 0)
%   % Segment:
%   vals_1 = 1 * vals_1;
%   
%   % Links:
%   vals_2 = (-1) * vals_2; 
% 
%   %% Equation:  
%   ineq_ix = ineq_ix + 1;
%   A_cols{ineq_ix}    = cat(1, ineq_cols_1, ineq_cols_2);
%   A_values{ineq_ix}  = cat(1, vals_1, vals_2);  
%   rhs_values(ineq_ix)     = 0;  
  
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
nCols = n_total_variables;
nzmax = numel(lin_rows);

% lin_rows
% lin_cols
% lin_values
A_polytope    = sparse(double(lin_rows), double(lin_cols), lin_values,...
   double(nRows),double(nCols),nzmax);
rhs_polytope  = sparse(rhs_values);

% This LHS is -Inf because we have defined the constraint as \sum <= 0!!
% as opposed to, say, \sum <= M
lhs_polytope  = repmat(-Inf, numel(rhs_polytope), 1);

end

% 