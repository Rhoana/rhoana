function [T_prob, A_smooth, T_smooth, A_gc, T_gc] = getGraphCutConnectivityMatrizes2D(im, imProb, ...
    threshProb, mGray, mstd, cs, ms)

T_prob = buildT([imProb(:)], threshProb);

n = numel(im);

disp('*** A started ***');
param = struct();
param.sigma = 1; %was 1
[A_smooth,T_smooth] = buildA(im, 1, 1,'E_smooth', param);
param.membraneGrayValue = mGray;
param.sigma = mstd;
param.cs = cs;
param.ms = ms;

[A_gc,T_gc] = buildA_01_10_direction(1,imProb, 1, 1,'E_direction', param); 

 