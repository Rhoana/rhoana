%a = angle should be in rad
%function imRot = centeredRotate(im, a)
function imRot = centeredRotate(im, a)
  a = double(a)/pi*180;
% %  im = max(max(im)) - im;
%   Trot = [cos(a) sin(a) 0; -sin(a) cos(a) 0; 0 0 1];
%   Ttrans = [1 0 0; 0 1 0; size(im,1)/2 size(im,2)/2 100];
%   imRot = transformImageFast(double(im), double(im), (inv(Ttrans) * Trot * (Ttrans)));
% %  imRot = max(max(imRot)) - imRot;

imRot = imrotate(im, a, 'bicubic','crop');