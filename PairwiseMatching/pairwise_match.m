function pairwise_match( input_block1_mat, abs_cube_from, side_from, input_block2_mat, abs_cube_to, side_to, output_mat, dicing_pars )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Change joining thresholds here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Join 1 (less joining)
auto_join_pixels = 20000; % Join anything above this many pixels overlap
minoverlap_pixels = 2000; % Consider joining all pairs over this many pixels overlap
minoverlap_dual_ratio = 0.7; % If both overlaps are above this then join
minoverlap_single_ratio = 0.9; % If either overlap is above this then join
    
% Join 2 (more joining)
% auto_join_pixels = 10000; % Join anything above this many pixels overlap
% minoverlap_pixels = 1000; % Consider joining all pairs over this many pixels overlap
% minoverlap_dual_ratio = 0.5; % If both overlaps are above this then join
% minoverlap_single_ratio = 0.8; % If either overlap is above this then join
    
disp(['Running pairwise matching']);

n_cubes_to_process = 2;

spt_M_per_cube    = cell(n_cubes_to_process,1);
i_s_L_sppst       = cell(n_cubes_to_process,1);

%Discover how many solutions will be processed
load_me                       = load(input_block1_mat, 'cliques_per_solution');
n_fusion_solutions            = numel(load_me.cliques_per_solution);
fusion_solutions_to_process   = 1:n_fusion_solutions;

win_link_As = cell(n_fusion_solutions,1);
win_link_Bs = cell(n_fusion_solutions,1);
split_label_As = cell(n_fusion_solutions,1);
split_label_Bs = cell(n_fusion_solutions,1);

add_cliques_As = cell(n_fusion_solutions,1);
add_cliques_Bs = cell(n_fusion_solutions,1);
add_cliques_links_As = cell(n_fusion_solutions,1);
add_cliques_links_Bs = cell(n_fusion_solutions,1);

for fusion_solution_to_use =  fusion_solutions_to_process
    
    fprintf(1, '\n----- Solution %d -----\n', fusion_solution_to_use);

    %Load cube 1 (from cube)
    fprintf(1, 'Loading: %s.\n', input_block1_mat);
    load_cube = load(input_block1_mat);

    cliques_per_solution = load_cube.cliques_per_solution;
    clique_map_from = load_cube.spt_M(cliques_per_solution{fusion_solution_to_use},:);
    label_vol_from = load_cube.i_s_L_sppst;

    clear load_cube;

    %Load cube 2 (to cube)
    fprintf(1, 'Loading: %s.\n', input_block2_mat);
    load_cube = load(input_block2_mat);

    cliques_per_solution = load_cube.cliques_per_solution;
    clique_map_to = load_cube.spt_M(cliques_per_solution{fusion_solution_to_use},:);
    label_vol_to = load_cube.i_s_L_sppst;

    clear load_cube;
    
    
    %% Volumetric joining - CMOR
    %% Overlap volumes
    disp('Computing overlap volumes between cubes');
    overlap_from = get_cube_overlap(label_vol_from, abs_cube_from, side_from, dicing_pars);%
    overlap_to = get_cube_overlap(label_vol_to, abs_cube_to, side_to, dicing_pars);
    
    %Link lables between volumes
    valid = overlap_from(:) ~= 0 & overlap_to(:) ~= 0;
    
    matches = unique([overlap_from(valid) overlap_to(valid)], 'rows');
    
    fprintf(1, 'Found %d potential matching 2d labels.\n', size(matches,1));
    
    stats_from = regionprops(overlap_from, 'Area', 'PixelIdxList');
    stats_to = regionprops(overlap_to, 'Area', 'PixelIdxList');
    
    edge_volume = get_overlap_edges(size(overlap_from), class(overlap_from), side_from);
    stats_edges = regionprops(edge_volume, 'PixelIdxList');
    
    match_sizes = zeros(size(matches,1), 1);
    edge_overlap_from = zeros(size(matches,1), 1);
    edge_overlap_to = zeros(size(matches,1), 1);
    
    for matchi = 1:size(matches,1)
        label_from = matches(matchi, 1);
        label_to = matches(matchi, 2);
        intersection_index = intersect(stats_from(label_from).PixelIdxList, stats_to(label_to).PixelIdxList);
        match_sizes(matchi) = size(intersection_index, 1);
        edge_overlap_from(matchi) = size(intersect(intersection_index, stats_edges(1).PixelIdxList), 1);
        edge_overlap_to(matchi) = size(intersect(intersection_index, stats_edges(2).PixelIdxList), 1);
    end
    
    %Put these into a sparse matrix
    %label_area_from = sparse(double(matches(:,1)), double(matches(:,2)), [stats_from(matches(:,1)).Area]);
    %label_area_to = sparse(double(matches(:,1)), double(matches(:,2)), [stats_to(matches(:,2)).Area]);
    label_matches = sparse(double(matches(:,1)), double(matches(:,2)), match_sizes);
    label_edge_overlap_from = sparse(double(matches(:,1)), double(matches(:,2)), edge_overlap_from);
    label_edge_overlap_to = sparse(double(matches(:,1)), double(matches(:,2)), edge_overlap_to);
        
    %Check each clique for overlaps
    clique_from_max = size(clique_map_from, 1);
    clique_to_max = size(clique_map_to, 1);
    
    %clique_nconn_from = zeros(clique_from_max, 1);
    %clique_nconn_to = zeros(clique_to_max, 1);
    clique_area_from = zeros(clique_from_max, 1);
    clique_area_to = zeros(clique_to_max, 1);
    clique_intersection = sparse([], [], [], clique_from_max, clique_to_max, size(matches,1));
    clique_intersection_edge_from = sparse([], [], [], clique_from_max, clique_to_max, size(matches,1));
    clique_intersection_edge_to = sparse([], [], [], clique_from_max, clique_to_max, size(matches,1));
    
    for matchi = 1:size(matches,1)
        label_from = matches(matchi, 1);
        label_to = matches(matchi, 2);
        
        from_clique_ix = find(clique_map_from(:,label_from));
        to_clique_ix = find(clique_map_to(:,label_to));
        
        if (length(from_clique_ix) > 1 || length(to_clique_ix) > 1)
            disp('WARNING: Found a label mapped for more than one clique!');
            from_clique_ix = from_clique_ix(1);
            to_clique_ix = to_clique_ix(1);
        end
        
        %clique_nconn_from(from_clique_ix) = clique_nconn_from(from_clique_ix) + 1;
        %clique_nconn_to(to_clique_ix) = clique_nconn_to(to_clique_ix) + 1;
        
        clique_area_from(from_clique_ix) = clique_area_from(from_clique_ix) + stats_from(label_from).Area;
        clique_area_to(to_clique_ix) = clique_area_to(to_clique_ix) + stats_to(label_to).Area;
        
        clique_intersection(from_clique_ix, to_clique_ix) = clique_intersection(from_clique_ix, to_clique_ix) + ...
            label_matches(label_from, label_to);

        clique_intersection_edge_from(from_clique_ix, to_clique_ix) = clique_intersection_edge_from(from_clique_ix, to_clique_ix) + ...
            label_edge_overlap_from(label_from, label_to);

        clique_intersection_edge_to(from_clique_ix, to_clique_ix) = clique_intersection_edge_to(from_clique_ix, to_clique_ix) + ...
            label_edge_overlap_to(label_from, label_to);

    end
    
    %Decide to join OR SPLIT based on overlap ratio and size
    
    [linkrow linkcol] = find(clique_intersection);
    n_potential_links = length(linkrow);
    
    fprintf(1, 'Simplified to %d potential matching 3d cliques.\n', n_potential_links);

    %linkstats_nconn_from = zeros(n_potential_links, 1);
    %linkstats_nconn_to = zeros(n_potential_links, 1);
    linkstats_area_from = zeros(n_potential_links, 1);
    linkstats_area_to = zeros(n_potential_links, 1);
    linkstats_area_overlap = zeros(n_potential_links, 1);
    linkstats_ratio_from = zeros(n_potential_links, 1);
    linkstats_ratio_to = zeros(n_potential_links, 1);
    
    for linki = 1:n_potential_links
        
        clique_from_i = linkrow(linki);
        clique_to_i = linkcol(linki);
        
        intersect_area = full(clique_intersection(clique_from_i, clique_to_i));
        
        %linkstats_nconn_from(linki) = clique_nconn_from(clique_from_i);
        %linkstats_nconn_to(linki) = clique_nconn_to(clique_to_i);
        
        linkstats_area_from(linki) = clique_area_from(clique_from_i);
        linkstats_area_to(linki) = clique_area_to(clique_to_i);
        
        linkstats_area_overlap(linki) = intersect_area;
        
        linkstats_ratio_from(linki) = intersect_area / clique_area_from(clique_from_i);
        linkstats_ratio_to(linki) = intersect_area / clique_area_to(clique_to_i);
        
%         fprintf(1, 'Link of size %d pixels found between clique %d (from, area=%d, %1.4f%%) and clique %d (to, area=%d, %1.4f%%).\n', ...
%             intersect_area, ...
%             clique_from_i, clique_area_from(clique_from_i), intersect_area / clique_area_from(clique_from_i) * 100, ...
%             clique_to_i, clique_area_to(clique_to_i), intersect_area / clique_area_to(clique_to_i) * 100);

    end
    
    %Take all 1-to-1 label links
    %Not needed - handled by ratio links if overlap is large enough
    %goodlinks = linkstats_nconn_from == 1 && linkstats_nconn_to == 1;
    
    %Take all links over the auto join volume
    goodlinks = linkstats_area_overlap >= auto_join_pixels;
    
    %Take all links over pixel and ratio thresholds
    goodlinks = goodlinks | ...
        (linkstats_area_overlap >= minoverlap_pixels & ...
        ((linkstats_ratio_from >= minoverlap_dual_ratio & ...
        linkstats_ratio_to >= minoverlap_dual_ratio) | ...
        (linkstats_ratio_from >= minoverlap_single_ratio | ...
        linkstats_ratio_to >= minoverlap_single_ratio)));
    
    %Also link one-sided volumes to the adjoining volume and
    %disconnect the far clique from other labels (edit clique map)
    %(Add one new clique to far side with only these labels and join the
    %new clique to the winning clique on the near side...)
    
    add_cliques_lbl_A = cell(1,0);
    add_cliques_linkto_A = cell(1,0);
    add_cliques_lbl_B = cell(1,0);
    add_cliques_linkto_B = cell(1,0);
    
    for linki = 1:n_potential_links
        if ~goodlinks(linki)
            %Check for one-side joins
            clique_from_i = linkrow(linki);
            clique_to_i = linkcol(linki);
            
            from_edge_pixels = clique_intersection_edge_from(clique_from_i, clique_to_i);
            to_edge_pixels = clique_intersection_edge_to(clique_from_i, clique_to_i);
            
            %Greedy join
            if from_edge_pixels > to_edge_pixels
            %Conservative join
            %if from_edge_pixels > 0 && to_edge_pixels == 0
                
                %from clique wins - split to clique
                new_clique_to = clique_map_to(clique_to_i,:);
                label_list = find(new_clique_to);
                %If any of these labels also belong to the to clique
                for cli = 1:length(label_list)
                    label_to = label_list(cli);
                    labels_from = matches(matches(:,2)==label_to,1);
                    if any(clique_map_from(clique_from_i,labels_from))
                        clique_map_to(clique_to_i,label_list(cli)) = false;
                    end
                end
                new_clique_to(clique_map_to(clique_to_i,:)~=0) = false;
                add_cliques_lbl_B{end+1} = new_clique_to;
                add_cliques_linkto_B{end+1} = clique_from_i;

                %knit this new clique into the result vectors
                %clique_to_max = clique_to_max + 1;
                %linkrow(end+1) = clique_from_i;
                %linkcol(end+1) = clique_to_max;
                %goodlinks(end+1) = true;
                
            %Greedy join
            else
            %Conservative join
            %elseif from_edge_pixels == 0 && to_edge_pixels > 0
                
                %to clique wins - split from clique
                new_clique_from = clique_map_from(clique_from_i,:);
                label_list = find(new_clique_from);
                %If any of these labels also belong to the to clique
                for cli = 1:length(label_list)
                    label_from = label_list(cli);
                    labels_to = matches(matches(:,1)==label_from,2);
                    if any(clique_map_to(clique_to_i,labels_to))
                        clique_map_from(clique_from_i,label_list(cli)) = false;
                    end
                end
                new_clique_from(clique_map_from(clique_from_i,:)~=0) = false;
                add_cliques_lbl_A{end+1} = new_clique_from;
                add_cliques_linkto_A{end+1} = clique_to_i;
                
                %knit this new clique into the result vectors
                %clique_from_max = clique_from_max + 1;
                %linkrow(end+1) = clique_from_max;
                %linkcol(end+1) = clique_to_i;
                %goodlinks(end+1) = true;
                
            end
        end
    end
    
    %TODO: Check new cliques for connected components...
    
    fprintf(1, 'Found %d good links and %d bad links.\n', sum(goodlinks), sum(~goodlinks));
    fprintf(1, 'Split %d cliques from cube %d.\n', length(add_cliques_lbl_A), abs_cube_from);
    fprintf(1, 'Split %d cliques from cube %d.\n', length(add_cliques_lbl_B), abs_cube_to);
    
    win_link_As{fusion_solution_to_use} = linkrow(goodlinks);
    win_link_Bs{fusion_solution_to_use} = linkcol(goodlinks);
    
    add_cliques_As{fusion_solution_to_use} = add_cliques_lbl_A;
    add_cliques_Bs{fusion_solution_to_use} = add_cliques_lbl_B;
    add_cliques_links_As{fusion_solution_to_use} = add_cliques_linkto_A;
    add_cliques_links_Bs{fusion_solution_to_use} = add_cliques_linkto_B;
    
    %Determine labels to mark as split-areas (for later joining to one adjicent clique)
    badlinks = sparse(linkrow(~goodlinks), linkcol(~goodlinks), ones(sum(~goodlinks),1), clique_from_max, clique_to_max);

    split_from = sparse(size(clique_map_from, 2), 1);
    split_to = sparse(size(clique_map_to, 2), 1);
    
    for matchi = 1:size(matches,1)
        label_from = matches(matchi, 1);
        label_to = matches(matchi, 2);
        
        from_clique_ix = find(clique_map_from(:,label_from));
        to_clique_ix = find(clique_map_to(:,label_to));
        
        if (length(from_clique_ix) > 1 || length(to_clique_ix) > 1)
            disp('WARNING: Found a label mapped for more than one clique!');
            from_clique_ix = from_clique_ix(1);
            to_clique_ix = to_clique_ix(1);
        end
        
        if badlinks(from_clique_ix, to_clique_ix)
            split_from(label_from) = true;
            split_to(label_to) = true;
        end
                
    end
    
    split_label_As{fusion_solution_to_use} = find(split_from);
    split_label_Bs{fusion_solution_to_use} = find(split_to);
    
    abs_cube_A = abs_cube_from;
    abs_cube_B = abs_cube_to;
    
end

disp(' ');
disp(['Saving to disk ...']);
t_measure = tic;

save(output_mat, 'win_link_As', 'win_link_Bs', 'split_label_As', 'split_label_Bs', 'abs_cube_A', 'abs_cube_B', ...
    'add_cliques_As', 'add_cliques_Bs', 'add_cliques_links_As', 'add_cliques_links_Bs', '-v7.3');

disp(['Saved in ' secs2hms(toc(t_measure))]);

disp(' ');
disp('All set!')

end