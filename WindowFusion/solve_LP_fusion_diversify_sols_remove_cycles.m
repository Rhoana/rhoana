function [solution_LP_fusion, cplex_obj] = solve_LP_fusion_diversify_sols_remove_cycles( ...
    clique_rep_data, ...
    link_rep_data, ...
    clique_stats,...
    link_stats, ...
    n_solutions_to_generate, ...
    n_frames_to_process, ...
    solv_options, ...
    polytope_pars, ...
    clq_cost_cntrnts, ...
    fusion_vs_nested_fusion)

n_solutions_to_generate = 200;

%% Get the clique and link variables
n_total_cliques   = clique_rep_data.spt_maps.n_total_cliques;
n_links           = link_rep_data.n_links;

vars_cliques              = uint32(1:n_total_cliques)';
last_ix                   = vars_cliques(end);

% One for each 3D link
vars_links                 = uint32((last_ix+1):(last_ix+1 + n_links-1))';
last_ix  = vars_links(end);
n_variables = vars_links(end);

%% Compute costs
disp(' ');
alpha_indices = [];
clq_cost_cntrnts.link_max_ovlap = 0;
clq_cost_cntrnts.link_min_ovlap = 0;

LP_costs  = compute_LP_costs(...
    clique_rep_data, ...
    link_rep_data, ...
    clique_stats, ...
    link_stats, ...
    clq_cost_cntrnts, ...
    alpha_indices);
% struct2var(LP_costs);

%%
clique_ix_to_frame_ix = clique_rep_data.spt_maps.clique_ix_to_frame_ix;
clique_sizes         = clique_stats.clique_sizes;

% In rows
costs = LP_costs.thetas;
costs = -costs;

n_aux_variables         = n_solutions_to_generate;
n_variables             = numel([vars_cliques; vars_links]);
n_total_variables       = n_variables + n_aux_variables;

n_clique_variables      = numel(vars_cliques);
n_variables_with_costs  = numel(costs);

vars_mult_solutions = [(n_variables+1):(n_variables+1+n_aux_variables-1)]';
%%


links_As = link_rep_data.links_As;
links_Bs = link_rep_data.links_Bs;
all_bckwd_links_per_clique = link_rep_data.all_bckwd_links_per_clique;
all_fwd_links_per_clique   = link_rep_data.all_fwd_links_per_clique;

LP_mtdata = var2struct(...
    vars_cliques, ....
    vars_links, ...
    n_total_variables, ...
    n_variables_with_costs, ...
    links_As, ...
    links_Bs, ...
    n_total_cliques, ...
    n_links, ...
    clq_cost_cntrnts, ...
    all_bckwd_links_per_clique,...
    all_fwd_links_per_clique, 'no');
clear links_As links_Bs all_bckwd_links_per_clique all_fwd_links_per_clique
%%





cnstr_ix = 0;
%% OVERLAP
% Matrix to prevent superposition:
% (EACH superpos sppixel can pick at most 1)
% A_polytope_2 = sparse([],[],[],double(n_ineqs),double(n_variables));

cnstr_ix = cnstr_ix +1;

sp_eqs_ovlap            = clique_rep_data.spt_mf_M;
sp_eqs_ovlap            = sp_eqs_ovlap';

[i,j,s]                 = find(sp_eqs_ovlap);
nRows                   = max(i);
nCols                   = n_total_variables;

add_A{cnstr_ix}         = sparse(i,j,s,nRows,double(nCols));
add_lhs_rows{cnstr_ix}  = repmat(0, nRows, 1);
add_rhs_rows{cnstr_ix}  = sparse(ones(nRows,1));
%% ONE-TO-ONE
cnstr_ix = cnstr_ix +1;

[ add_lhs_rows{cnstr_ix}, ...
    add_A{cnstr_ix}, ...
    add_rhs_rows{cnstr_ix} ] = build_polytope_one_to_one(LP_mtdata);
%%

add_A        = cell2mat(add_A');
add_lhs_rows = cell2mat(add_lhs_rows');
add_rhs_rows = cell2mat(add_rhs_rows');

%%
% Additional control variables?
% A_additional_colums =  sparse(size(add_A,1), n_aux_variables);
% add_A = cat(2, add_A, A_additional_colums);
%%
t_solver = tic;

disp('Building the cplex object ...');
% This assumes that you have CPLEX installed on your system and on the
% MATLAB path. CPLEX is the option by default.
%    echo on all;
clear cplex_obj;

cplex_obj = Cplex();

% Pre-solver? (which includes the aggregator) (indicator variable)
% cplex_obj.Param.preprocessing.presolve.Cur = 0;

% Silent!
% cplex_obj.DisplayFunc = [];
% otherwise DisplayFunc = @disp ?? (not sure how to rebuild a function

f = zeros(n_total_variables, 1);
f(1:n_variables_with_costs) = costs;

%%
if ~isempty(find(isnan(f), 1))
    disp(' -----------------------------');
    disp(' There are NaN values in f! ' );
    disp(' -----------------------------');
end
%%
cplex_obj.Model.obj   = f;
cplex_obj.Model.A     = add_A;
cplex_obj.Model.lhs   = add_lhs_rows;
cplex_obj.Model.rhs   = add_rhs_rows;
cplex_obj.Model.sense = 'minimize';
cplex_obj.Model.ctype = repmat('B',n_total_variables,1);

% cplex_obj.Model.obj   = randn(n_variables, 1);
% Variable bounds:
cplex_obj.Model.lb    = zeros(n_total_variables,1);
cplex_obj.Model.ub    = ones(n_total_variables,1);

% No verbosity
% cplex_obj.DisplayFunc= [];

cplex_obj.Param.clocktype.Cur = 2;
cplex_obj.Param.threads.Cur = 1;
cplex_obj.Param.timelimit.Cur = solv_options.time_limit; %1 hour
% cplex_obj.Param.timelimit.Cur = 60; %1 hour


% Find top solution
disp(' ');
disp('Solving for the top solution ...');
t_solve = tic;

%% MIP start:
cplex_obj.Param.advance.Cur = 1;
cplex_obj.MipStart          = [];

MipStart           = cplex_obj.MipStart;
cplex_obj.MipStart = [];

next_MIP = numel(MipStart) + 1;
MipStart(next_MIP).name          = 'all_zero';
MipStart(next_MIP).effortlevel   = 1;
MipStart(next_MIP).x             = zeros(n_total_variables,1);
MipStart(next_MIP).xindices      = 1:n_total_variables;

cplex_obj.MipStart = MipStart;
%%
cplex_obj.solve();
disp(['Done ' secs2hms(toc(t_solve))]);

x = logical(round(cplex_obj.Solution.x));

which_clique_sol = x(vars_cliques); % Binary vector of the n_2D_cliques
which_link_sol = x(vars_links);   % Binary vector of the n_2D_cliques

raw_sol           = x;
raw_obj_val      = cplex_obj.Solution.objval;
cliques_win_ix   = find(which_clique_sol);
links_win_ix     = find(which_link_sol);

sol_one_to_one  = ...
    var2struct( ...
    links_win_ix, ...
    cliques_win_ix, ...
    raw_sol, ...
    raw_obj_val, ...
    'no');


% control_frame_ix = round(n_frames_to_process/2);

% At least we have one solution (from solving)
%% TEMPORAL:
solution_LP_fusion = compute_multiple_solutions_with_branching( ...
    clique_rep_data, link_rep_data, clique_stats, link_stats, ...
    sol_one_to_one, ...
    LP_mtdata, LP_costs,  ...
    cplex_obj, solv_options, polytope_pars, fusion_vs_nested_fusion);


n_solutions = numel(solution_LP_fusion.raw_sols);
link_costs  = cell(n_solutions, 1);
for sol_ix = 1:n_solutions
    x = solution_LP_fusion.raw_sols{sol_ix};
    link_costs{sol_ix} = LP_costs.thetas_links(x(vars_links));
end

solution_LP_fusion.link_costs = link_costs;

% % 1) Fix one-to-one segments and links to one
% % 2) Limit the # of extra links that can be added
% % 3) Allow branching
% % 4) Add limits on area loss ...
%
% % # amvr TO_DO :
% % Generate multiple solutions
% nested_fusion = false;
% disp(' ');
% disp(['Letting it branch ...']);
% redefine_polytope_and_let_it_branch;
%    % Returns sol_let_it_branch
% %%
% % Get rid of cycles:
% % if ispc keyboard;end
% %%
%
% % initial_cplex_obj  = cplex_obj;
% % sol_cycle_cuts     = cut_and_solve_remove_cycles(  ...
% %                         initial_cplex_obj,         ...
% %                         vars_cliques,              ...
% %                         vars_links,                ...
% %                         link_rep_data,             ...
% %                         sol_let_it_branch,               ...
% %                         n_total_variables,         ...
% %                         solv_options.n_max_cuts);
% % solutions.cycle_cuts    = sol_cycle_cuts;
% %%
%
% all_cliques_win_ix      = {sol_one_to_one.cliques_win_ix};
% all_links_win_ix        = {sol_one_to_one.links_win_ix};
% raw_obj_vals            = {sol_one_to_one.raw_obj_val};
% raw_sols                = {sol_one_to_one.raw_sol};
%
%
% sol_to_concatenate      = sol_let_it_branch;
% all_cliques_win_ix      = cat(1, all_cliques_win_ix, {sol_to_concatenate.cliques_win_ix}');
% all_links_win_ix        = cat(1, all_links_win_ix, {sol_to_concatenate.links_win_ix}');
% raw_obj_vals            = cat(1, raw_obj_vals, {sol_to_concatenate.raw_obj_val}');
% raw_sols                = cat(1, raw_sols, {sol_to_concatenate.raw_sol}');
% n_solutions_retrieved   = 2;
%
%
% solutions.one_to_one    = sol_one_to_one;
% solutions.let_it_branch = sol_let_it_branch;
%
% %%
% % raw_sols = cellfun(@(x) x', raw_sols, 'UniformOutput', false);
% % solutions         = cell2mat(raw_sols);
%
% solution_LP_fusion = var2struct(...
%                         all_cliques_win_ix, ...
%                         all_links_win_ix, ...
%                         raw_obj_vals,...
%                         raw_sols,...
%                         n_solutions_retrieved, ...
%                         'no');

disp(['Done in ' secs2hms(toc(t_solver))]);

end
