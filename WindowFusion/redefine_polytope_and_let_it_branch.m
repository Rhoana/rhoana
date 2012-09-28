function  [add_A, add_lhs_rows, add_rhs_rows] = redefine_polytope_and_let_it_branch( ...
      sol_one_to_one, ...
      clique_rep_data, link_rep_data, ...
      LP_mtdata, polytope_pars)
   
%% Dynamic fusion-and-cut. Two steps:
% Repeat until we obtain a desired number of solutions

%% add_dynamic_cut_1 (only done once)
% 1) Find activated links
% 2) Fix them to 1
% 3) Allow an X number of extra links to be activated
% 4) Alllow branching

% The steps # 3 and 4 produce a polytope

%% add_dynamic_cut_2 (done until no more cycles)
% 1) Find cycles
% 2) Remove cycles

% This is how we index solutions:
      %    which_sol = x(vars_links); 
      %    clq_3D_win_ix = find(which_sol);
      %    links_As(clq_3D_win_ix)
      %    links_Bs(clq_3D_win_ix)
      % 
      %    links_As(clq_3D_win_ix) = [];
      %    links_Bs(clq_3D_win_ix) = [];
      %    links_to_remove_lin_ix = sub2ind(size(solution_graph), links_As,links_Bs);
      %    solution_graph(links_to_remove_lin_ix) = 0;
%

% This function should output:
% add_A
% add_lhs_rows
% add_rhs_rows
struct2var(LP_mtdata);
add_A_cut = {};
add_lhs_row_cut = {};
add_rhs_row_cut = {};

% # amvr FIX_ME
% Do we want to add likns?
n_max_cuts      = 5; 
N_new_links     = Inf;
cnstr_ix        = 1;

n_total_cliques   = clique_rep_data.spt_maps.n_total_cliques;

%% 1) Preventing overlap
sp_eqs_ovlap     = clique_rep_data.spt_mf_M;
sp_eqs_ovlap     = sp_eqs_ovlap';

% n_eqs_ovlap = # of cliques in the superposition 
n_eqs_ovlap     = size(sp_eqs_ovlap,1);

% One variable per 2D clique 
[i,j,s]            = find(sp_eqs_ovlap);
nRows               = max(i);
nCols               = n_total_variables;
add_A_cut{cnstr_ix} = sparse(i,j,s,nRows,double(nCols));
add_lhs_row_cut{cnstr_ix} = repmat(0, nRows, 1);
add_rhs_row_cut{cnstr_ix} = sparse(ones(nRows,1));

%% If a clique is activated, one link must be activated too!


%% Find activated links and force them to one

   cnstr_ix = cnstr_ix + 1;
   links_win_ix = sol_one_to_one.links_win_ix;
   vars_links_that_won_in_one_to_one   = vars_links(links_win_ix);
   n_vars_links_that_won_in_one_to_one = numel(links_win_ix);

   % Fill in the sparse matrix:
   rows_temp = 1:n_vars_links_that_won_in_one_to_one;
   rows_temp = rows_temp';
   cols_temp = double(vars_links_that_won_in_one_to_one);
   vals_temp = ones(n_vars_links_that_won_in_one_to_one,1);

   nRows = n_vars_links_that_won_in_one_to_one;
   nCols = n_total_variables;
   nzmax = n_vars_links_that_won_in_one_to_one;

   add_A_fix_links_one_to_one = sparse(rows_temp, cols_temp, vals_temp, ...
                                 double(nRows),double(nCols), nzmax);


   add_A_cut{cnstr_ix}         = add_A_fix_links_one_to_one;
   add_lhs_row_cut{cnstr_ix}   = ones(n_vars_links_that_won_in_one_to_one,1);
   add_rhs_row_cut{cnstr_ix}   = add_lhs_row_cut{cnstr_ix};

if 0
   %% Allow an X number of extra links to be activated MAX
   cnstr_ix = cnstr_ix + 1;

   N_current_links = n_vars_links_that_won_in_one_to_one;


   % Fill in the sparse matrix:
   rows_temp = ones(n_3D_variables, 1);
   cols_temp = double(vars_links);
   vals_temp = ones(n_3D_variables,1);


   nRows     = 1;
   nCols     = n_total_variables;
   nzmax     = n_3D_variables;

   add_A_limit_N_links        = sparse(rows_temp, cols_temp, vals_temp, ...
                                 double(nRows),double(nCols), nzmax);
   disp(['We had ' num2str(N_current_links)  ' links']);
   disp(['We allow for ' num2str(N_new_links)  ' extra links']);

   add_A_cut{cnstr_ix}               = add_A_limit_N_links;
   add_lhs_row_cut{cnstr_ix}         = N_current_links;
   add_rhs_row_cut{cnstr_ix}         = N_current_links + N_new_links;

end

%% Allow for branching:
% Dynamic cuts are for branching and removing cycles !!
cnstr_ix = cnstr_ix + 1;

[ add_lhs_row_cut{cnstr_ix}, ...
     add_A_cut{cnstr_ix}, ...
     add_rhs_row_cut{cnstr_ix} ] = build_polytope_allow_branching(LP_mtdata);

%%  Limit area loss:
if polytope_pars.label_loss
   cnstr_ix = cnstr_ix + 1;
   if nested_fusion == false
      [ add_lhs_row_cut{cnstr_ix}, ...
      add_A_cut{cnstr_ix}, ...
      add_rhs_row_cut{cnstr_ix} ] = ...  
         build_polytope_control_label_loss(...
            vars_cliques, ....
            vars_links, ...
            n_total_variables, ...
            links_As, ...
            links_Bs, ...
            all_bckwd_links_per_clique,...
            all_fwd_links_per_clique, ...
            n_total_cliques, ...
            n_frames_to_process, ...
            clique_sizes, ...
            clique_ix_to_frame_ix, ...
            clq_cost_cntrnts);
   else

      [ add_lhs_row_cut{cnstr_ix}, ...
      add_A_cut{cnstr_ix}, ...
      add_rhs_row_cut{cnstr_ix} ] = ...  
      build_polytope_control_label_loss_nested(...  
               vars_cliques, ....
               vars_links, ...
               n_total_variables, ...
               links_As, ...
               links_Bs, ...
               all_bckwd_links_per_clique,...
               all_fwd_links_per_clique, ...
               n_total_cliques, ...
               n_cubes_to_process, ...
               clique_2D_iface_sizes, ...
               clique_ix_to_cube_ix, ...
               clq_cost_cntrnts);   
   end
end
   
   
%%
      add_A        = cell2mat(add_A_cut');
      add_lhs_rows = cell2mat(add_lhs_row_cut');
      add_rhs_rows = cell2mat(add_rhs_row_cut');
%%      RE-SOLVE ALLOWING FOR BRANCHING:
% cplex_obj.A = [];


% clear cplex_obj;
% cplex_obj = Cplex();
