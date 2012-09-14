%computes value for direction propagation in graphcut
%p.sigma = sigma for exp function
%p.cs = contextsize for orientation
%p.ms = membranesize for orientation
%ASSUMPTION: arg1 and arg2 all are in the same neighbourhood direction
%function vals = E_direction(im,arg1, arg2, p)
function vals = E_direction10(im,arg1, arg2, p)
  %arg2 is used to estimate neighbourhood angle that is required
% $$$     im = double(im);
% $$$   [r1,c1] = ind2sub(size(im),arg1(1));
% $$$   [r2,c2] = ind2sub(size(im),arg2(1));
% $$$   b = [r1,c1] - [r2,c2];
% $$$   if b(1) < 0
% $$$     b = -b;
% $$$   end
% $$$   a = [0,1];
% $$$   angle = -acos(a(1)*b(1)+a(2)*b(2)/(sqrt(a(1)^2+a(2)^2)*sqrt(b(1)^2+b(2)^2)));
% $$$   
% $$$   d = zeros(p.cs);
% $$$   d(:,round(p.cs/2)-round(p.ms/2):round(p.cs/2)+round(p.ms/2)) = 1;
% $$$   d = centeredRotate(d,angle);
% $$$  
% $$$   orient = normxcorr2_mex(double(d), 1-double(im), 'same');
% $$$   vals = (orient(arg1).^2).* (1/(p.sigma*sqrt(2*pi))*exp(-0.5*im(arg1).^2 / p.sigma^2));

    
   vals = zeros(1,size(arg1,1));

