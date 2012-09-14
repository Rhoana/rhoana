%function [eig1, eig2, angle1, angle2, v11, v12, v21, v22] = eigImg(im)
function [eig1, eig2, angle1, angle2, v11, v12, v21, v22] = eigImg(im)
[gx,gy] = gradient(double(im));
[gxx,gxy] = gradient(gx);
[gxy,gyy] = gradient(gy);

clear gx;
clear gy;

eig1 = zeros(size(im));
eig2 = zeros(size(im));
angle1 = zeros(size(im));
angle2 = zeros(size(im));
v11 = zeros(size(im));
v12 = zeros(size(im));
v21 = zeros(size(im));
v22 = zeros(size(im));

N = version;

% if (sum(N(1:3)=='7.1')==3)
%   for i=1:length(im(:))
%     e = [gxx(i) gxy(i); gxy(i) gyy(i)];
%     [v,e] = eig(e);
%     eig1(i) = e(1,1);
%     eig2(i) = e(2,2);
%     angle1(i) = atan2(v(1,1), v(2,1));
%     angle2(i) = atan2(v(1,2), v(2,2));
%     v11(i) = v(1,1); v12(i) = v(2,1);
%     v21(i) = v(1,2); v22(i) = v(2,2);
%   end
% else
parfor i=1:length(im(:))
    e = [gxx(i) gxy(i); gxy(i) gyy(i)];
    [v,e] = eig(e);
    eig1(i) = e(1,1);
    eig2(i) = e(2,2);
    angle1(i) = atan2(v(1,1), v(2,1));
    angle2(i) = atan2(v(1,2), v(2,2));
    v11(i) = v(1,1); v12(i) = v(2,1);
    v21(i) = v(1,2); v22(i) = v(2,2);
end
%end

