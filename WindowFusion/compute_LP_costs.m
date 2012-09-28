function [LP_costs] = compute_LP_costs(...
    clique_rep_data, ...
    link_rep_data, ...
    clique_stats, ...
    link_stats, ...
    clq_cost_cntrnts,...
    alpha_indices)


%%
% One link covering the same area of three links should have the equivalent strength
% of the three links.

% One segment covering the same area of three segments should have the
% equivalent strength

% We want a bit of an over-segmentation


%%


%    clq_cost_cntrnts.size_compensation_factor = 0.8;
%    clq_cost_cntrnts.flg_size_calibration = true;


link_max_ovlap_thresh          =  clq_cost_cntrnts.link_max_ovlap;
link_min_ovlap_thresh          =  clq_cost_cntrnts.link_min_ovlap;

do_not_connect_theta       =  clq_cost_cntrnts.do_not_connect_theta;


n_total_cliques            = clique_rep_data.spt_maps.n_total_cliques;
n_links                    = link_rep_data.n_links;

links_As                   = link_rep_data.links_As;
links_Bs                   = link_rep_data.links_Bs;

links_bckwd_overlap     = link_stats.links_bckwd_overlap;
links_fwd_overlap       = link_stats.links_fwd_overlap;

mu                         = clq_cost_cntrnts.flow_disp_filt.mu;
std_deviation              = clq_cost_cntrnts.flow_disp_filt.std_deviation;
clique_boosting_factor     = clq_cost_cntrnts.clique_boosting_factor;

%% Size calibration:
flg_size_calibration    = clq_cost_cntrnts.flg_size_calibration;
if isfield(clq_cost_cntrnts, 'size_compensation_factor'),       size_compensation_factor = clq_cost_cntrnts.size_compensation_factor;
else       size_compensation_factor = 1;    end

fh_size_compensation = @(x) x.^size_compensation_factor;


%% CLIQUES:
% Get calibrations (size). This is important to make sure that
% small and large segments have the same probability of being selected
% in the output.

clique_boosting = ones(n_total_cliques, 1);
if ~isempty(alpha_indices)
    clique_boosting(alpha_indices) = clique_boosting_factor;
end

clique_sizes                  = clique_stats.clique_sizes;

thetas_cliques_size_calib     = clique_sizes;
thetas_cliques_lklhoods       = zeros(n_total_cliques,1);%-0.00001*


if flg_size_calibration,
    thetas_cliques             = fh_size_compensation(thetas_cliques_size_calib);
else
    thetas_cliques             = thetas_cliques_lklhoods;
end

%    thetas_cliques                = -0.001 * ones(n_total_cliques,1);


%% LINKS:

% We boost links according to clique boosting
link_boosting              = clique_boosting(links_As(:)) + clique_boosting(links_Bs(:));
links_max_overlap          = max(links_fwd_overlap, links_bckwd_overlap);
links_min_overlap          = min(links_fwd_overlap, links_bckwd_overlap);

%    link_sizes                 = links_min_overlap; %clique_sizes(links_As(:)) + clique_sizes(links_Bs(:));
link_sizes                 = min(clique_sizes(links_As), clique_sizes(links_Bs));
thetas_links_size_calib    = link_sizes;
thetas_links_lklhoods      = links_max_overlap;

if flg_size_calibration
    thetas_links            = thetas_links_lklhoods.* ...
        fh_size_compensation(thetas_links_size_calib);
else
    thetas_links            = thetas_links_lklhoods;
end

% Which links are OK to connect:
% This has to be done after size calibration!
which_ok  = links_max_overlap>=link_max_ovlap_thresh;
thetas_links(~which_ok) =  -do_not_connect_theta;

which_ok  = links_min_overlap>=link_min_ovlap_thresh;
thetas_links(~which_ok) =  -do_not_connect_theta;

% Technically only if alpha_indices were not empty
% thetas_links                    = thetas_links.*link_boosting;

% max_screen_ratio = clq_cost_cntrnts.disp_limits.max_screen_ratio;
% thetas_links(link_to_disp_mag>max(comp_mtdata.width, comp_mtdata.width)/max_screen_ratio) = -Inf;

%
%    thetas_2D_cliques(alpha_indices)    = fh_size_compensation(1*ones(numel(alpha_indices),1).*thetas_2D_cliques_size_calib(alpha_indices));
thetas                              = [thetas_cliques; thetas_links];


LP_costs = var2struct(...
    clique_sizes, ...
    link_sizes, ...
    clique_boosting, ...
    link_boosting, ...
    thetas_cliques, ...
    thetas_links, ...
    thetas_cliques_lklhoods, ...
    thetas_links_lklhoods, ...
    thetas, ...
    'no');

end


