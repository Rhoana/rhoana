function [clq_2D_data, i_stacks] = ...
                      get_2D_cliques(i_thin_hard_bdries, i_s_sppst)

disp('Building spt_mf_M');
   % nR, nC,nZ = size
%% Get superposition map and i_L_tesells  

% i_thin_hard_bdries(row_ix, col_ix, frame_ix, tessellation_ix)

nR = size(i_thin_hard_bdries, 1);
nC = size(i_thin_hard_bdries, 2);
nZ = size(i_thin_hard_bdries, 3);

n_tessellations_per_frame = zeros(nZ, 1, 'uint32');
for frame_ix = 1:nZ
  n_tessellations_per_frame(frame_ix) = size(i_thin_hard_bdries, 4);
end
    
i_L_tesells = cell(nZ, 1);

n_tessell_ccs   = cell(nZ,1);
SE = strel('diamond',1);
disp('Collecting superpositions ...');
for frame_ix = 1:nZ
    n_tessell_ccs{frame_ix}                             = zeros(n_tessellations_per_frame(frame_ix),1, 'uint32');
    
    i_sppst                      = squeeze(i_s_sppst(:,:,frame_ix));
    for tessell_ix = 1:n_tessellations_per_frame(frame_ix)               
        i_thin_hard_bdry                        = squeeze(i_thin_hard_bdries(:,:, frame_ix, tessell_ix));
        CC                                      = bwconncomp(~i_thin_hard_bdry, 4);
        n_tessell_ccs{frame_ix}(tessell_ix)     = CC.NumObjects;
        i_L_tesells{frame_ix}(:,:,tessell_ix)   = uint32(labelmatrix(CC));
    end
    % We make sure we avoid the problem with potentially overlapping
    % cliques in the output (imagiine four 2D segmentes overlapping but
    % shifted one pixel between each)

    for tessell_ix = 1:n_tessellations_per_frame(frame_ix)
       i_L_tessell = i_L_tesells{frame_ix}(:,:,tessell_ix);
       i_L_tessell(i_sppst) = 0;
       i_L_tesells{frame_ix}(:,:,tessell_ix) = i_L_tessell;
    end
end

% We might have to thicken the thin boundaries ...

%% Superposition matrices for fusion - one per frame and per tessellation
% These matrices will have to be merged later.
sps_tessell_segs_to_superpsts_per_frame = cell(nZ,1);

n_superpst_ccs                          = zeros(nZ, 1, 'uint32');
for frame_ix = 1:nZ
    % Because different frames might have a different # of tessellations
    sps_tessell_segs_to_superpsts_per_frame{frame_ix}   = cell(n_tessellations_per_frame(frame_ix),1);
end

%% Get sps_tessell_segs_to_superpsts_per_frame
% Input: 
  % i_s_sppst
  % [nZ nR nC] (can be inferred from i_s_sppst)
  
%%  
i_s_L_sppst = zeros(nR, nC, nZ, 'uint32');

disp('Building the clique matrices per frame');

tess_labels_lost_per_frame_and_tess = cell(nZ, 1);

clique_to_cc_lbl = cell(nZ, 1);
for frame_ix = 1:nZ
    disp(['Frame ' num2str(frame_ix)]);
   
    % Labeling of the superposition maps
    CC                            = bwconncomp(~(squeeze(i_s_sppst(:,:,frame_ix))), 4);
    n_superpst_ccs(frame_ix)      = CC.NumObjects;
    i_L_superpst                  = labelmatrix(CC);
    
    % For the output!
    i_s_L_sppst(:,:, frame_ix)  = uint32(i_L_superpst);
    
    tess_labels_lost_per_frame_and_tess{frame_ix} = cell(n_tessellations_per_frame(frame_ix),1);

    for tessell_ix = 1:n_tessellations_per_frame(frame_ix)        
        i_L_tessell                    = squeeze(i_L_tesells{frame_ix}(:,:,tessell_ix));
        
        sp_this_tessell_segs_to_superpst = sparse([],[],[], ...
            double(n_tessell_ccs{frame_ix}(tessell_ix)),...
            double(n_superpst_ccs(frame_ix)));
         
         which_tess_labels_lost = false(n_tessell_ccs{frame_ix}(tessell_ix),1);
         
         % Get the superposition matrix:
         label_grouping    = accumarray(i_L_tessell(:)+1, i_L_superpst(:)+1, [], @(x) {x});   
         label_mapping     = cellfun(@(x) faster_unique(x(:)), label_grouping,           'UniformOutput', false);
         label_mapping     = cellfun(@(x) x(x ~= 1), label_mapping,      'UniformOutput', false);
         label_mapping(1)  = [];
         label_mapping     = cellfun(@(x) x-1, label_mapping,            'UniformOutput', false);


         lin_cols    = label_mapping;
         nR          = numel(label_mapping);
         lin_rows    = num2cell([1:nR]');
         lin_rows    = cell2mat(cellfun(@(x,y) repmat(x, numel(y),1), lin_rows, lin_cols, 'UniformOutput', false));
         which_tess_labels_lost = cellfun(@(x) isempty(x), lin_cols, 'UniformOutput', true);
         lin_cols(which_tess_labels_lost) = [];
         lin_cols    =  cell2mat(lin_cols);
         
         n_nnz       = numel(lin_cols);
         
         % nC          = max(lin_cols); % This may not work, since it could
         % be that one of the tessellations missed some superposition
         % labels (e.g. because no segment from the tessellation touches
         % such superposition segments). The # of columns of the matrix
         % should always be fixed to the # of superposition segments!
         % So we use max(i_L_superpst(:)) instead.
         nC          = max(i_L_superpst(:));
         lin_vals    = ones(n_nnz, 1);

         sp_this_tessell_segs_to_superpst    = sparse(double(lin_rows), double(lin_cols), lin_vals,...
                                                      double(nR),double(nC),n_nnz);
         
        tess_labels_lost_per_frame_and_tess{frame_ix}{tessell_ix}       = find(which_tess_labels_lost);
        a = sp_this_tessell_segs_to_superpst;
        
        % Remove empty rows (labels lost...)
	     b=a(sum(a,2)~=0,:);
        sp_this_tessell_segs_to_superpst = b;
        
        sps_tessell_segs_to_superpsts_per_frame{frame_ix}{tessell_ix}   = sp_this_tessell_segs_to_superpst;
    end
    
   %% We rebuild sps_tessell_segs_to_superpsts_per_frame{frame_ix}
   % Without repetitions!   
   n_cliques_per_tessellation = cellfun(@(x) size(x, 1), sps_tessell_segs_to_superpsts_per_frame{frame_ix});
   clique_to_tessell_all_tess = [];
   clique_to_cc_lbl{frame_ix} = cell(n_tessellations_per_frame(frame_ix), 1);
   clique_to_cc_lbl_all_tess = [];
   for tessell_ix = 1:n_tessellations_per_frame(frame_ix)   
      clique_to_tessell_all_tess = [clique_to_tessell_all_tess; repmat(tessell_ix, n_cliques_per_tessellation(tessell_ix), 1)];
      
      [r c] = find(sps_tessell_segs_to_superpsts_per_frame{frame_ix}{tessell_ix});
      if ~isempty(r)
            % FASTER UNIQUE:
               lbl_array = r;
               lbl_array = lbl_array(:);
               lbl_array = sort(lbl_array);
               d = diff(lbl_array);
               d = d ~=0;
               d = [true; d];
               u_vals_diff = lbl_array(d);          
      else      
         u_vals_diff = [];
      end
      
      clique_to_cc_lbl{frame_ix}{tessell_ix}  =  u_vals_diff;
      if isempty(u_vals_diff)
%          if ispc 
%             keyboard
%          end
      end
      clique_to_cc_lbl_all_tess = [clique_to_cc_lbl_all_tess; clique_to_cc_lbl{frame_ix}{tessell_ix}];
   end
   
   all_sps_this_frame = cell2mat(sps_tessell_segs_to_superpsts_per_frame{frame_ix});
   nR = size(all_sps_this_frame,1);
   [which_are_unique] = find_unique_rows(all_sps_this_frame, true(nR,1));

   all_sps_this_frame_no_reps                   = all_sps_this_frame(which_are_unique,:);
   clique_to_tessell_all_tess_no_reps           = clique_to_tessell_all_tess(which_are_unique);
   clique_to_cc_lbl_all_tess_no_reps            = clique_to_cc_lbl_all_tess(which_are_unique);
   
   %% We rebuild: sps_tessell_segs_to_superpsts_per_frame
   sps_tessell_segs_to_superpsts_per_frame_with_reps = sps_tessell_segs_to_superpsts_per_frame;
   for tessell_ix = 1:n_tessellations_per_frame(frame_ix)   
      %% Careful, we cannot lose superposition labels!
       sps_tessell_segs_to_superpsts_per_frame{frame_ix}{tessell_ix} = ...
          all_sps_this_frame_no_reps(clique_to_tessell_all_tess_no_reps==tessell_ix,:);         
   end

   %% Find the labels in each tessellation that were lost because of redundancy within the frame:
   cc_lbls_sps_tessell{frame_ix} = cell(n_tessellations_per_frame(frame_ix), 1);
   for tessell_ix = 1:n_tessellations_per_frame(frame_ix)   
       cc_lbls_before_prunning = clique_to_cc_lbl_all_tess(clique_to_tessell_all_tess == tessell_ix);
       cc_lbls_after_prunning  = clique_to_cc_lbl_all_tess_no_reps(clique_to_tessell_all_tess_no_reps == tessell_ix);
       which_were_kept         = ismember(cc_lbls_before_prunning, cc_lbls_after_prunning);
       
       % We update clique_to_cc_lbl
       clique_to_cc_lbl{frame_ix}{tessell_ix} = sort(cc_lbls_after_prunning);
       
       ccl_lbls_that_corresond_to_repetitions = find(~which_were_kept);
      
       % We update tess_labels_lost_per_frame_and_tess       
       tess_labels_lost_per_frame_and_tess{frame_ix}{tessell_ix} = ...
          [tess_labels_lost_per_frame_and_tess{frame_ix}{tessell_ix}; ...
          ccl_lbls_that_corresond_to_repetitions];         
   end
end


% For debugging
   % sps_tessell_segs_to_superpsts_per_frame_with_reps
   
%% --------------------------------------------------------%%
   % This is how we encode each segment
   % sps_tessell_segs_to_superpsts_per_frame
%% --------------------------------------------------------%%
n_labels_lost_frame_and_tess    =  cell(nZ, 1);
n_useful_labels_frame_and_tess  =  cell(nZ, 1);

for frame_ix = 1:nZ
  n_labels_lost_frame_and_tess{frame_ix} = zeros(n_tessellations_per_frame(frame_ix),1, 'uint32');
  for tessell_ix = 1:n_tessellations_per_frame(frame_ix)       
    n_labels_lost_frame_and_tess{frame_ix}(tessell_ix) = uint32(numel(tess_labels_lost_per_frame_and_tess{frame_ix}{tessell_ix}));
  end
  n_useful_labels_frame_and_tess{frame_ix} = int32(n_tessell_ccs{frame_ix}) - int32(n_labels_lost_frame_and_tess{frame_ix});
end

n_labels_lost_per_frame_due_to_sppst_reppresentation = zeros(nZ,1);
for frame_ix = 1:nZ
  n_labels_lost_per_frame_due_to_sppst_reppresentation(frame_ix) = sum(n_labels_lost_frame_and_tess{frame_ix});
end

%% The next step is to group these matrices into a single matrix:

% We go from: 
    % sps_tessell_segs_to_superpsts_per_frame
% to:
    % all_M
% We collect the matrices per tessellation
spt_Ms_per_frame = cell(nZ,1);
for frame_ix = 1:nZ
   spt_Ms_this_frame = cell2mat(sps_tessell_segs_to_superpsts_per_frame{frame_ix });
   
   which_rows_to_consider = true(size(spt_Ms_this_frame,1),1);
   [which_rows_to_keep]    = find_unique_rows(spt_Ms_this_frame, which_rows_to_consider);
   
   spt_Ms_per_frame{frame_ix} = spt_Ms_this_frame(which_rows_to_keep,:);   
end

clear sps_tessell_segs_to_superpsts_per_frame;
[spt_mf_M, spt_maps] = build_spt_mf_M_and_maps(spt_Ms_per_frame);

%%
n_2D_cliques = size(spt_mf_M,1);

clq_2D_data = var2struct(...
        spt_mf_M, ...
        spt_maps, ...
        n_2D_cliques, ...
        'no');
i_stacks.i_s_L_sppst  = i_s_L_sppst;      

