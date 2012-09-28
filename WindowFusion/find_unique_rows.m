function [which_are_unique] = find_unique_rows(in_M, which_are_cnsdrd_for_insp)

    %set_default_value(2, 'which_are_cnsdrd_for_insp', true(size(in_M,1),1));
    
    n_inds                                          = size(in_M, 1);    
    
    considered_for_inspection_ixs                   = find(which_are_cnsdrd_for_insp);
    in_M_subset_in_which_to_check_for_repetitions   = in_M(which_are_cnsdrd_for_insp,:);
    
    %
    [~, unique_occurrences_ixs, ~]                  = unique(in_M_subset_in_which_to_check_for_repetitions, 'rows', 'first'); 

    which_are_repetitions                            = true(sum(which_are_cnsdrd_for_insp),1);
    which_are_repetitions(unique_occurrences_ixs)    = false; 

    % Indices to remove:
    keep_from_original_ixs                          = considered_for_inspection_ixs;
    keep_from_original_ixs(which_are_repetitions)    = [];

    which_are_unique                                = false(n_inds,1);
    which_are_unique(keep_from_original_ixs)        = true;

    % We also keep the ones that we didn't consider for inspection!
    which_are_unique                               = which_are_unique | ~which_are_cnsdrd_for_insp;
    
    % If we wish to report them as well
    % which_are_repetitions                          = ~which_are_uniques;

end