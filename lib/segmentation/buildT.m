%this function builds T edge weights for GraphCut
%s is represented by 1 , t is represented by 2
%dataFunc needs to exist in two versions dataFunc_0 and dataFunc_1
%function T = buildT(E0, E1)
function T = buildT(im, lambda)
n = numel(im);

t = (2*lambda - 2*(im(:)));
c = (t<0)+1;
T = sparse(1:numel(im),c,t,numel(im),2);
T(:,2) = -T(:,2);
