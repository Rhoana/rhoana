function window_fusion( input_seg_hdf5, global_block_idx, output_hdf5 )

%Add the lib folder to the path
d = fileparts(which(mfilename));
addpath(genpath(fullfile(d, '..', 'lib', 'fusion')));

%input_seg_hdf5 = 'ExampleDicedBlock.hdf5';
%output_mat = 'ExampleFusionOutput.mat';
%output_hdf5 = 'ExampleFusionOutput.hdf5';

%%%%% FUSION SETTINGS %%%%%

dwnSmp = 1;
ppgtion_depth = 0;
adj_type = 'norm_disp'; %large_disp;norm_disp


target_comp_bw_size       = 100;
n_solutions_to_generate   = 4;


solv_options.area_change = 0.75;
solv_options.time_limit  = 1800; % In seconds
solv_options.n_max_cuts  =  5;


polytope_pars.one_to_one = false;
polytope_pars.label_loss = false;


%% clq_cost_cntrnts
loss_type.percentage      = 1;
loss_type.absolute_pixel  = 2;
loss_type.which           = loss_type.absolute_pixel;
%
loss_limits.percentage_limit = 0.2;
loss_limits.absolute_pixel   = 100*100; %500 160 = ~ (100/8).^2 [i.e. (size/8).^2]
%
disp_limits.max_screen_ratio = 6; % Any links bigger disp is made 0
%
flow_disp_filt.mu             = 0;
flow_disp_filt.std_deviation  = 6;
%
flg_size_calibration             = true;
space_time_seg_area_loss_limit   = 0.2;

% size_compensation_factor
% 0.975 was the best for ac3 but joins too much on the 5k cube
% 0.9 is ok on the 5k cube
% 0.8 will reduce merge errors in favour of split errors (for Omni etc)
size_compensation_factor         = 0.9;

% These settings are for branching: 
% link_n_costs: number of branching settings to try
% link_max_ovlap_thresh_range: overlap proportion threshold(s) for branch with larger overlap
% link_min_ovlap_thresh_range: overlap proportion threshold(s) for branch with smaller overlap
% Settings to disable branching
link_n_costs                     = 0;
link_max_ovlap_thresh_range      = [];
link_min_ovlap_thresh_range      = [];
% Settings for calculating three branching solutions
%link_n_costs                     = 3;
%link_max_ovlap_thresh_range      = [0.9 0.95 0.99];
% Values below work well for ac3 but join too much for the 5k cube
%link_min_ovlap_thresh_range      = [0.15 0.2 0.25];
% Values below will join less and might provide reasonable results for the 5k cube (untested)
%link_min_ovlap_thresh_range      = [0.35 0.4 0.45];

do_not_connect_theta             = 0.2;
n_max_branches                   = Inf;
clique_boosting_factor           = 2;

clq_cost_cntrnts = var2struct(loss_limits, ...
    loss_type, ...
    disp_limits, ...
    flow_disp_filt, ...
    flg_size_calibration, ...
    space_time_seg_area_loss_limit, ...
    size_compensation_factor, ...
    link_n_costs, ...
    link_min_ovlap_thresh_range, ...
    link_max_ovlap_thresh_range, ...
    n_max_branches, ...
    do_not_connect_theta, ...
    clique_boosting_factor, ...
    'clear');


fusion_vs_nested_fusion = 'fusion';


seg_rendering_args.flg_region_filling                    = false;
seg_rendering_args.flg_crack_inpainting                  = false;


tess_pars.tess_smallest_seg_area_proportion_for_region_filling  = 8;


%%%%% END SETTINGS %%%%%


t_fusion_total = tic;

% Read in the segmentations
% Fusion expects logicals with 1 for boundary and 0 for non-boundary pixels

i_thin_hard_bdries = hdf5read(input_seg_hdf5, '/cubesegs') < 1;

n_frames_to_process  = size(i_thin_hard_bdries, 3);

fprintf(1, 'Processing %d frames.', n_frames_to_process);

flg_upsampling  = false;

area_to_clean        = round(0.5*dwnSmp);
i_s_L_sppst_re_tess   = get_i_s_L_sppst_from_tessellations( ...
    i_thin_hard_bdries, ...
    area_to_clean); % or -> cube_

% startProfile
[clique_rep_data, i_stacks] = get_2D_cliques(i_thin_hard_bdries,i_s_L_sppst_re_tess == 0);
% stopProfile

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Use 30 here!! %%
%%%%%%%%%%%%%%%%%%%%%%%%%
% % Rebuild clique_rep_data
% clique_rep_data = rebuild_spt_mf_M_on_new_i_sppst_stack(...
%             clique_rep_data, ...
%             i_s_L_sppst_original, ...
%             i_s_L_sppst_re_tess, ...
%             n_frames_to_process);
% i_stacks.i_s_L_sppst  = i_s_L_sppst_re_tess;

%% Remove repetitions and empty cliques:
[clique_rep_data.spt_mf_M, clique_rep_data.spt_maps] = ...
    spt_remove_repetitions_and_empty_cliques(clique_rep_data.spt_mf_M, clique_rep_data.spt_maps);
%%

clear i_s_L_sppst_original
clear i_s_L_sppst_re_tess

% [clique_rep_data.spt_mf_M, clique_rep_data.spt_maps] = remove_empty_cliques (...
%    clique_rep_data.spt_mf_M, clique_rep_data.spt_maps, n_frames_to_process);

%% Propagation:

% Collect the optical flows (they should have been computed before if ppgtion depth ~= 0)
if ppgtion_depth == 0
    uv_flows = [];
else
    uv_flows = load_opt_flows_from_disk(...
        p_to_opt_flow_folder, ...
        frames_to_process, ...
        cube_mtdata); % or -> cube_
    
    matrix_size = n_frames_to_process;
    ppgation_matrix = spdiags(ones(matrix_size,2*ppgtion_depth+1),...
        -ppgtion_depth:ppgtion_depth,matrix_size,matrix_size);
    %%
    
    clique_rep_data = get_ppgted_spt_mf_M_from_original_spt_mf_M(...
        clique_rep_data, ...
        i_stacks.i_s_L_sppst, ...
        ppgation_matrix, ...
        i_stacks.i_s_L_sppst, ...
        uv_flows, ...
        n_frames_to_process);
end
%%
[clique_rep_data.spt_mf_M, clique_rep_data.spt_maps] = ...
    spt_remove_repetitions_and_empty_cliques(clique_rep_data.spt_mf_M, clique_rep_data.spt_maps);

%%

%%
disp (' ');
disp('Obtain pre-segmentations');
alpha_indices_ochs         = [];
alpha_indices_irfan_essa   = [];
alpha_indices_ucm          = [];
if 0 % If we need to rescue
    
    if bdry_flags.ochs
        [alpha_indices_ochs, clique_rep_data.spt_mf_M,  clique_rep_data.spt_maps] = ...
            find_my_cliques_from_i_s_tessellation(...
            clique_rep_data.spt_mf_M, clique_rep_data.spt_maps, i_stacks.i_s_L_sppst, i_thin_hard_bdries_ochs);
    end
    
    
    if bdry_flags.ochs
        [alpha_indices_irfan_essa, clique_rep_data.spt_mf_M,  clique_rep_data.spt_maps] = ...
            find_my_cliques_from_i_s_tessellation(...
            clique_rep_data.spt_mf_M, clique_rep_data.spt_maps, i_stacks.i_s_L_sppst, i_thin_hard_bdries_irfan_essa);
    end
    
    
    if bdry_flags.gPb
        [alpha_indices_ucm, clique_rep_data.spt_mf_M,  clique_rep_data.spt_maps] = ...
            find_my_cliques_from_i_s_tessellation(...
            clique_rep_data.spt_mf_M, clique_rep_data.spt_maps, i_stacks.i_s_L_sppst, i_thin_hard_bdries_ucm);
    end
end


[clique_rep_data.spt_mf_M, clique_rep_data.spt_maps] = ...
    spt_remove_repetitions_and_empty_cliques(clique_rep_data.spt_mf_M, clique_rep_data.spt_maps);


%% Get the 3D adjacency matrix
disp(' ');
disp('Determining frame-to-frame adjacency using sppst-to-sppst multiplication');
link_rep_data                                  = get_3D_adjacency(clique_rep_data, i_stacks, adj_type, size(i_thin_hard_bdries)); % or -> cube_
%
% If we want to add branching to the mix...
% [spt_mf_M, spt_maps] = add_splits_to_spt_mf_M(clique_rep_data, link_rep_data, n_frames_to_process);

%% Binary bitmaps
disp(' ');
disp('Computing the binary bitmaps for each face of each cube');

[~, i_s_bw_non_cent, comp_mtdata]              = get_i_s_bw_all_2D_cliques(clique_rep_data, i_stacks, target_comp_bw_size);

%% Compute displaced bitmaps
% i_s_bw_non_cent_displaced = i_s_bw_non_cent;


%% Compute clique and link statistics
disp(' ');
disp('Determining clique and link statistics ...');


clique_stats                 = get_2D_cliques_stats(clique_rep_data, i_stacks, uv_flows, dwnSmp);
%%
link_stats                   = get_3D_links_stats(...
    clique_rep_data, ...
    link_rep_data,  ...
    i_s_bw_non_cent, ...
    i_s_bw_non_cent); % i_s_bw_non_cent_displaced
%% Compute clique stats

%[clique_stats, link_stats]    = normalize_stats(clique_stats, link_stats, cube_mtdata, comp_mtdata); % or -> cube_
clique_stats.spt_maps.clique_sizes     = clique_stats.clique_sizes / (dwnSmp.^2);


%%   Solve and render i_mats
% [solutions_LP_fusion, ~]          = solve_LP_fusion_remove_sols( ...
%          polytope, clique_rep_data.spt_maps, LP_costs, n_solutions_to_generate, n_frames_to_process, solv_options);
%%
% [solutions_LP_fusion, ~]          = solve_LP_fusion_remove_top_sol( ...
%          polytope, clique_rep_data.spt_maps, LP_costs, n_solutions_to_generate, n_frames_to_process, solv_options);


% [solutions_LP_fusion, ~]          = solve_LP_fusion_diversify_sols( ...
%          polytope, clique_rep_data.spt_maps, LP_costs, n_solutions_to_generate, n_frames_to_process, solv_options, ...
%          clique_rep_data, ...
%          link_rep_data, ...
%          i_stacks);

%%
%%
% [solutions_LP_fusion, ~]          = solve_LP_fusion_diversify_sols_vble( ...
%          polytope, clique_rep_data.spt_maps, LP_costs, n_solutions_to_generate, n_frames_to_process, solv_options);



%% We need to output weights for CC.
%%
% Parse the solution to determine connected components and build image
% stacks with the results

%%
%%
clear i_s_bw_non_cent;
[solutions_LP_fusion] = solve_LP_fusion_diversify_sols_remove_cycles( ...
    clique_rep_data, ...
    link_rep_data, ...
    clique_stats,...
    link_stats, ...
    n_solutions_to_generate, ...
    n_frames_to_process, ...
    solv_options, ...
    polytope_pars, ...
    clq_cost_cntrnts, ...
    fusion_vs_nested_fusion);


%%
all_cliques_win_ix   = solutions_LP_fusion.all_cliques_win_ix;
all_links_win_ix     = solutions_LP_fusion.all_links_win_ix;

n_solutions          = numel(all_cliques_win_ix);
i_s_L_solutions      = cell(n_solutions,1);
%
disp(' '); tic;
disp(['Building stack of image results for ' num2str(n_solutions) ' solutions']);
cube_clique_costs = cell(n_solutions,1);
for solution_ix = 1:n_solutions
    disp(['Solution: ' num2str(solution_ix)]);
    cliques_win_ix = all_cliques_win_ix{solution_ix};
    links_win_ix   = all_links_win_ix{solution_ix};
    
    [ i_s_L_solutions{solution_ix},  cube_clique_costs{solution_ix}, ~] = build_image_stack_from_fusion_solution( ...
        cliques_win_ix, ...
        links_win_ix, ...
        clique_rep_data, ...
        link_rep_data, ...
        i_stacks, ...
        size(i_thin_hard_bdries), ...
        seg_rendering_args.flg_region_filling, ...
        seg_rendering_args.flg_crack_inpainting, ...
        tess_pars.tess_smallest_seg_area_proportion_for_region_filling, ...
        solutions_LP_fusion.link_costs{solution_ix});
end

% disp(['Done with building the stacks of image results in ' secs2hms(toc)]);
% 
% %%
% disp(' '); tic;
% disp('Saving independent solutions.');
% 
% 
% disp('We build the fusion cube matrix and save it to disk ...'); tic
% [spt_M, i_s_L_sppst, cliques_per_solution] = ...
%     build_spt_M_cube_no_clique_props(i_s_L_solutions, i_stacks.i_s_L_sppst);
% 
% 
% n_cliques_per_solution = cellfun(@(x) numel(x), cliques_per_solution);
% n_solutions            = numel(cliques_per_solution);
% disp(['Done with building and saving the cube matrix in ' secs2hms(toc)]);
% 
% t_clique_stats = tic;
% disp('Computing clique stats');
% 
% clique_stats_cube  = get_cube_clique_stats(spt_M, i_s_L_sppst);
% disp(['Done with clique stats in ' secs2hms(toc(t_clique_stats))]);
% 
% raw_obj_vals = solutions_LP_fusion.raw_obj_vals;

% save(output_mat, ...
%       'n_solutions', ...
%       'cube_clique_costs', ...
%       'n_cliques_per_solution', ...
%       'cliques_per_solution', ...      
%       'i_s_L_solutions', ...
%       'spt_M', ...
%       'i_s_L_sppst', ...
%       'raw_obj_vals', ...
%       'clique_stats_cube');

%%  spt_M_nonzeros = find(spt_M);
%%  spt_M_size = size(spt_M);
%%  
%%  hdf5write(output_hdf5, ...
%%      '/n_solutions', n_solutions, ...
%%      '/cube_clique_costs', cube_clique_costs, ...
%%      '/n_cliques_per_solution', n_cliques_per_solution, ...
%%      '/cliques_per_solution', cliques_per_solution, ...
%%      '/i_s_L_solutions', i_s_L_solutions, ...
%%      '/spt_M_nonzeros', spt_M_nonzeros, ...
%%      '/spt_M_size', spt_M_size, ...
%%      '/i_s_L_sppst', i_s_L_sppst, ...
%%      '/raw_obj_values', solutions_LP_fusion.raw_obj_vals, ...
%%      '/clique_stats_cube', clique_stats_cube);
%%  
%%  

% Map to dense volume and add global block idx in high bits (but only in nonzero labels)
labeled_volume = uint64(i_s_L_solutions{1});
nz = find(labeled_volume);
labeled_volume(nz) = (uint64(2) ^ 32) * uint64(str2num(global_block_idx)) + uint64(labeled_volume(nz));

% avoid writing partial files
temp_hdf5 = [output_hdf5, '_partial'];
if exist(temp_hdf5, 'file'),
  delete(temp_hdf5);
end
h5create(temp_hdf5, '/labels', [Inf, Inf, Inf], 'DataType', 'uint64', 'ChunkSize', [64, 64, 4], 'Deflate', 9, 'Shuffle', true);
h5write(temp_hdf5, '/labels', labeled_volume, [1, 1, 1], size(labeled_volume));
movefile(temp_hdf5, output_hdf5);


disp(' ');
disp(['All set with ' mfilename '! Total time: ' secs2hms(toc(t_fusion_total))]);
%%
