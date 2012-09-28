function clq_stats = get_cube_clique_stats(spt_M_this_cube, i_s_L_sppst)

reg_stats   = regionprops(i_s_L_sppst, 'Area');
sppst_lbl_areas = [reg_stats.Area];

n_cliques_this_cube        = size(spt_M_this_cube,1);
clique_2D_side_sizes       = zeros(n_cliques_this_cube,6);
clique_internal_sizes      = zeros(n_cliques_this_cube,1);

for  this_clique_ix = 1:n_cliques_this_cube
    
    this_clique_sppst_ix = find(spt_M_this_cube(this_clique_ix,:));
    
    % Internal
    % Get the size of each 2D segment
    clique_internal_sizes(this_clique_ix) = sum(sppst_lbl_areas(this_clique_sppst_ix));
    
end

clq_stats = var2struct(clique_internal_sizes, clique_2D_side_sizes, 'no');
