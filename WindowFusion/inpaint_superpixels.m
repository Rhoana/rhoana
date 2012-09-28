function i_L_new_final = inpaint_superpixels(i_L, dist_th)

% set_default_value(2, 'start_new_labels_from_value', 1);
% set_default_value(3, 'single_or_CC_new_colors', 'm');
% set_default_value(4, 'dist_th', 3);
% set_default_value(5,'flg_fix_bdries', false);

i_L_zero = i_L == 0;
[i_D , L] = bwdist(~i_L_zero);

i_bw_pixels_to_inpaint       = (i_D <= dist_th) & i_L_zero;
% i_bw_pixels_not_to_inpaint   = (i_D > dist_th) & i_L_zero;

lin_indices_closest = L(i_bw_pixels_to_inpaint);
i_L_new = i_L;
i_L_new(i_bw_pixels_to_inpaint) = i_L(lin_indices_closest);

% 
% if flg_fix_bdries == true
%    i_L_new = fix_missing_boundaries(i_L_new);
% end

% 
% i_new_coloring = uint32(labelmatrix(bwconncomp(i_bw_pixels_not_to_inpaint)));
% 
% % If coloring by CC
% if strcmpi(single_or_CC_new_colors,'cc')
% i_new_coloring(i_new_coloring ~=0) = i_new_coloring(i_new_coloring ~=0) + (start_new_labels_from_value -1);
% else % If coloring by same value: (i.e. single_or_multiple_new_colors == 's'
% i_new_coloring(i_new_coloring ~=0) = start_new_labels_from_value;
% end
% 
% if any(any(i_L_new & i_new_coloring))
%    disp('Problems running inpainting!');
% end
%%
% Add colors:
% i_L_new_final = i_L_new + i_new_coloring;

% Don't add colors:
i_L_new_final = i_L_new;
% i_L_new_final = fix_missing_boundaries(i_L_new_final);