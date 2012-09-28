function i_output = thin_and_recolor_image(i_input)
   % This only works for tessellations where regions are perfectly enclosed
   % by black pixels! It will not recolor images where regions are not
   % enclosed by dark boundaries. For that, we need an erosion-oriented
   % relabeling. Note that accumarray is the way to do this though in both
   % cases!
   
   i_retess = i_input;

   i_tess = i_retess == 0;
   i_tess = bwdist(~i_tess);
   i_tess = watershed(i_tess, 4);
   i_tess = i_tess == 0;   
   i_tess(1:end,1)   = 1;   i_tess(1,1:end) = 1;
   i_tess(1:end,end) = 1;   i_tess(end,1:end) = 1;      

   i_input_bdries_thin = i_tess;
   i_retess = labelmatrix(bwconncomp(~i_input_bdries_thin, 4));

   label_grouping     = accumarray(i_retess(:)+1, i_input(:)+1, [], @(x) {x});   
   mapping_of_retess_labels_left  = cellfun(@(x) faster_unique(x(:)), label_grouping,           'UniformOutput', false);
   mapping_of_retess_labels_left  = cellfun(@(x) x(x ~= 1), mapping_of_retess_labels_left,      'UniformOutput', false);
   mapping_of_retess_labels_left(1) = [];
   mapping_of_retess_labels_left  = cellfun(@(x) x-1, mapping_of_retess_labels_left,            'UniformOutput', false);
   which_are_empty                = cellfun(@(x) isempty(x), mapping_of_retess_labels_left,    'UniformOutput', true);

   % Otherwise this would give errors! (we can't assign something to
   % nothing)
   if any(which_are_empty) 
      mapping_of_retess_labels_left(which_are_empty) = ...
         cellfun(@(x) cast(0, class(i_input)), mapping_of_retess_labels_left(which_are_empty), 'UniformOutput', false);
   end     
   mapping_of_retess_labels_left  = cell2mat(mapping_of_retess_labels_left);

   % Recolor!
   i_zero = i_retess == 0;
   i_retess(i_zero) = max(i_retess(:));
   i_retess = mapping_of_retess_labels_left(i_retess);
   i_retess(i_zero)  = 0;
   i_output = i_retess;
end