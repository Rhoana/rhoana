function solution_LP_fusion = compute_multiple_solutions_with_branching( ...
    clique_rep_data, ...
    link_rep_data, ...
    clique_stats, ...
    link_stats, ...
    sol_one_to_one, ...
    LP_mtdata, ...
    LP_costs,  ...
    cplex_obj, ...
    solv_options, ...
    polytope_pars, ...
    fusion_vs_nested_fusion)

% # amvr TO_DO :
% Generate multiple solutions
% LP_costs
% LP_mtdata;
% sol_one_to_one
% cplex_obj
% solv_options
% polytope_pars

vars_links              = LP_mtdata.vars_links;
vars_cliques            = LP_mtdata.vars_cliques;
n_variables_with_costs  = LP_mtdata.n_variables_with_costs;
n_total_variables       = LP_mtdata.n_total_variables;


nested_fusion = false;
disp(' ');
disp(['Letting it branch ...']);

all_cliques_win_ix      = {sol_one_to_one.cliques_win_ix};
all_links_win_ix        = {sol_one_to_one.links_win_ix};
raw_obj_vals            = {sol_one_to_one.raw_obj_val};
raw_sols                = {sol_one_to_one.raw_sol};

n_costs = LP_mtdata.clq_cost_cntrnts.link_n_costs;
solutions.let_it_branch = cell(n_costs, 1);

[add_A, add_lhs_rows, add_rhs_rows] = redefine_polytope_and_let_it_branch(...
    sol_one_to_one, clique_rep_data, link_rep_data, LP_mtdata, polytope_pars);

for cost_ix = 1:n_costs
    disp(' ');
    disp(['Solving for costs ' num2str(cost_ix)]);
    LP_mtdata.clq_cost_cntrnts.link_max_ovlap  = LP_mtdata.clq_cost_cntrnts.link_max_ovlap_thresh_range(cost_ix);
    LP_mtdata.clq_cost_cntrnts.link_min_ovlap  = LP_mtdata.clq_cost_cntrnts.link_min_ovlap_thresh_range(cost_ix);
    alpha_indices = [];
    if strcmp(fusion_vs_nested_fusion, 'fusion')
        LP_costs  = compute_LP_costs(...
            clique_rep_data, ...
            link_rep_data, ...
            clique_stats, ...
            link_stats, ...
            LP_mtdata.clq_cost_cntrnts, ...
            alpha_indices);
    else % Nested fusion
        LP_costs = compute_LP_costs_nested(...
            clique_rep_data, link_rep_data, ...
            clique_stats, link_stats, ...
            LP_mtdata.clq_cost_cntrnts);
    end
    
    costs = LP_costs.thetas;
    costs = -costs;
    
    f = zeros(n_total_variables, 1);
    f(1:n_variables_with_costs) = costs;
    
    cplex_obj.Model.obj   = f;
    cplex_obj.Model.A     = add_A ;
    cplex_obj.Model.lhs   = add_lhs_rows;
    cplex_obj.Model.rhs   = add_rhs_rows;
    cplex_obj.Model.lb    = repmat(0, n_total_variables,1);
    cplex_obj.Model.ub    = repmat(1, n_total_variables,1);
    
    cplex_obj.Model.sense = 'minimize';
    cplex_obj.Model.ctype = repmat('B',n_total_variables,1);
    cplex_obj.Param.advance.Cur   = 1;
    cplex_obj.Param.clocktype.Cur = 2;
    cplex_obj.Param.threads.Cur = 1;
    cplex_obj.Param.timelimit.Cur = solv_options.time_limit; %1 hour
    
    % cplex_obj.addRows(add_lhs_rows, add_A, add_rhs_rows);
    %%
    disp('Calling solve ...');
    cplex_obj.solve();
    
    %%
    disp(['Message from CPLEX: ' cplex_obj.Solution.statusstring]);
    disp(['-------------------------']);
    try
        x = logical(round(cplex_obj.Solution.x));
    catch me
        % If this happens, it is because we couldn't get as many solutions as
        % we originally wanted...
        disp('Error accessing optimal solution (x) from cplex object');
        n_solutions_retrieved = solution_ix;
        %    break;
    end
    
    which_clique_sol = x(vars_cliques); % Binary vector of the n_total_cliques
    which_link_sol = x(vars_links);   % Binary vector of the n_total_cliques
    
    % If polytope is empty...
    if ~any(which_clique_sol)
        n_solutions_retrieved = solution_ix;
        %    break;
    end
    
    raw_sol = x;
    raw_obj_val = cplex_obj.Solution.objval;
    
    cliques_win_ix  = find(which_clique_sol);
    links_win_ix    = find(which_link_sol);
    
    sol_let_it_branch = ...
        var2struct( ...
        links_win_ix, ...
        cliques_win_ix, ...
        raw_sol, ...
        raw_obj_val, ...
        add_A, ...
        'no');
    
    
    
    sol_to_concatenate      = sol_let_it_branch;
    all_cliques_win_ix      = cat(1, all_cliques_win_ix, {sol_to_concatenate.cliques_win_ix}');
    all_links_win_ix        = cat(1, all_links_win_ix, {sol_to_concatenate.links_win_ix}');
    raw_obj_vals            = cat(1, raw_obj_vals, {sol_to_concatenate.raw_obj_val}');
    raw_sols                = cat(1, raw_sols, {sol_to_concatenate.raw_sol}');
    
    solutions.let_it_branch{cost_ix} = sol_let_it_branch;
end


n_solutions_retrieved   = 1 + n_costs; %One to one + let it branch..

solutions.one_to_one    = sol_one_to_one;

solution_LP_fusion = var2struct(...
    all_cliques_win_ix, ...
    all_links_win_ix, ...
    raw_obj_vals,...
    raw_sols,...
    n_solutions_retrieved, ...
    'no');


