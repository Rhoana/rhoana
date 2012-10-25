function s = var2struct(varargin)

  % Last argument determines if I clear or not the variables
  opt = varargin{end}; 
  varargin(end) = [];
  
  names = arrayfun(@inputname,1:nargin-1,'UniformOutput',false);
  s = cell2struct(varargin,names,2);
 
  for name_ix = 1:numel(names)-1
    names{name_ix}(end+1) = ' ';
  end
  
  if strcmp(opt, 'clear');
    evalin('caller', ['clear ' cell2mat(names)]);
  end

end

