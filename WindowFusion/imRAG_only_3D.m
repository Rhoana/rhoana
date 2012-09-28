function varargout = imRAG_only_3D (img, varargin)

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
  varargout{1} = edges;
  
end