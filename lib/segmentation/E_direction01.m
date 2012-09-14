%computes value for direction propagation in graphcut
%p.sigma = sigma for exp function
%p.cs = contextsize for orientation
%p.ms = membranesize for orientation
%ASSUMPTION: arg1 and arg2 all are in the same neighbourhood direction
%function vals = E_direction(im,arg1, arg2, p)
function vals = E_direction01(im,arg1, arg2, p)
    im = double(im);
    orient = p.orientImg;
    distance = p.dist;
 
    %vals = (orient(arg1).^2).* (1/(p.sigma*sqrt(2*pi))*exp(-0.5*(im(arg1)-p.membraneGrayValue).^2 / p.sigma^2))*distance;
    %abs instead of ^2 and leave away weird normalization
    vals = abs(orient(arg1)).* exp(-0.5*(im(arg1)-p.membraneGrayValue).^2 / p.sigma^2)*distance;
    