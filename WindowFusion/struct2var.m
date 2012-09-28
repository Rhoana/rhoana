function struct2var(s)
  
%    cellfun(@(n,v) assignin('caller',n,v),fieldnames(s),struct2cell(s));

    sCell = struct2cell(s);
    fNames = fieldnames(s);
    for field_ix = 1:numel(fNames)
        assignin('caller', fNames{field_ix},sCell{field_ix});
    end
end