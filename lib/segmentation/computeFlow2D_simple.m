function [ims] = computeFlow2D_simple(im, T_prob, A_smooth, T_smooth, A_gc, T_gc)
T = T_prob + T_smooth + T_gc;
T = sparse(T);
clear T_prob
clear T_smooth
clear T_gc
clear T_na
clear T_flux

A = A_smooth + A_gc;
A = sparse(A);
clear A_smooth
clear A_gc
clear A_na

[f,l] = maxflow(A,T);

n = numel(im);
ims = double(reshape(l(1:n),size(im)));



