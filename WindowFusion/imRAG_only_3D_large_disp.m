function varargout = imRAG_only_3D_large_disp (img, varargin)

    dim = size(img);
    edges = [];
% compute matrix of absolute differences in the third direction
	diff3 = abs(diff(img, 1, 3));
	
	% find non zero values (region changes)
	[i1 i2 i3] = ind2sub(size(diff3), find(diff3));
	
	% delete values close to border
	i1 = i1(i3<dim(3)-1);
	i2 = i2(i3<dim(3)-1);
	i3 = i3(i3<dim(3)-1);
	
	% get values of consecutive changes
	val1 = diff3(sub2ind(size(diff3), i1, i2, i3));
	val2 = diff3(sub2ind(size(diff3), i1, i2, i3+1));
	
	% find changes separated with 2 pixels
	ind = find(val2 & val1~=val2);
	
   edges = [edges ; unique([val1(ind) val2(ind)], 'rows')];
   
   
   %% Large displacement

   i_pre  = sq(img(:,:,1));
   i_next = sq(img(:,:,3));
   rag_i_pre  = double(imRAG(i_pre));
   rag_i_next = double(imRAG(i_next));

   edges_from = edges(:,1);
   edges_to   = edges(:,2);

   rag_i_next_edges_from = [rag_i_next(:,1)];%rag_i_next(:,2)];
   rag_i_next_edges_to   = [rag_i_next(:,2)];%%;rag_i_next(:,1)];

%    edges_from = [edges(:,1); edges(:,2)];
%    edges_to   = [edges(:,2); edges(:,1)];

   nghboring_edges_on_next_from_lower  = arrayfun( @(x) rag_i_next_edges_to(rag_i_next_edges_from == x), edges_to, 'UniformOutput', false);
   nghboring_edges_on_next_from_higher = arrayfun( @(x) rag_i_next_edges_from(rag_i_next_edges_to == x), edges_to, 'UniformOutput', false);
   
   nghboring_edges_on_next = cellfun(@(x,y) cat(1,x,y), nghboring_edges_on_next_from_lower, nghboring_edges_on_next_from_higher, 'UniformOutput', false);

   extra_edges_from = num2cell(edges_from);
   extra_edges_from = cell2mat(cellfun(@(x,y) repmat(x, numel(y),1), extra_edges_from, nghboring_edges_on_next, 'UniformOutput', false));
   extra_edges_to   = cell2mat(nghboring_edges_on_next);

   new_edges = [extra_edges_from extra_edges_to];
   new_edges = sortrows(new_edges);
   which_are_unique = find_unique_rows(new_edges);
   new_edges = new_edges(which_are_unique, :);

   all_edges = [edges; new_edges]; 
   
   which_are_unique = find_unique_rows(all_edges);
   all_edges        = all_edges(which_are_unique, :);
   
   %%
   % Remember: extra_edges_to and extra_edges_to INCLUDE DUPLICATES!
   % extra_edges_to(extra_edges_from == 35)
   % ans =
   %          85.00
   %          87.00
   %          55.00
   %          86.00
   %          87.00
   %          55.00
   %          59.00
   %          83.00
   %          84.00

   % Remember: new_edges DOES NOT INCLUDE DUPLICATES!
   % new_edges(new_edges(:,1) == 1, :)
   % ans =
   %            1          45
   %            1          46
   %            1          66
   %            1          68
   %            1          78
   %            1          81
   %            1          87
   %            1          91
   %            1          92   
   
   varargout{1} = all_edges;
end