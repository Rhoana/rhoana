%     Copyright 2011 Seymour Knowles-Barley.
%
%     This file is part of Density2d.
% 
%     Density2d is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     Density2d is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with Density2d.  If not, see <http://www.gnu.org/licenses/>.


% creategabor generates a receptive field based on an input average image
% and a gaussian size
% img    : Input image
% sigma  : Gaussian size (s.d. in pixels)
% gamma  : Eliptical aspect ratio (1 = circular, 0.5 = tall, 2 = wide)
% gabor  : A matrix containing the (sometimes gabor-like) signal
function [g3] = creategabor3(img, sigma, in_radius, out_radius, gamma, angle, scale, curve)

r = in_radius;
w = in_radius*2+1;

[x,y]=meshgrid(-r:r,-r:r);
d = sqrt(x.^2 + gamma.^2*y.^2);
g3 = img .* normpdf(d, 0, sigma);
%Normalize
g3 = g3 ./ normpdf(0, 0, sigma);

if curve ~= 0
    %Curve the receptive field
    [gx,gy] = meshgrid(1:w+2);
    %gx = gx + (gy-r).^2/r*curve;
    circle_radius = r/curve;
    gx = gx + sqrt(circle_radius.^2 + (gy-(r+2)).^2) - circle_radius;
    gborder = zeros(w+2);
    gborder(2:w+1,2:w+1) = g3;
    gborder = interp2(gborder, gx, gy);
    g3 = gborder(2:w+1,2:w+1);
    g3(isnan(g3)) = 0;
end

if scale ~= 1
    %Scale the receptive field
    g3 = imresize(g3, scale);
    if scale < 1
        %pad with zeros
        gborder = zeros(w);
        sg = size(g3,1);
        offset = round((w-sg)/2);
        gborder(offset+1:offset+sg,offset+1:offset+sg) = g3;
        g3 = gborder;
    else
        mid = round((size(g3,1))/2);
        g3 = g3(mid-r:mid+r,mid-r:mid+r);
    end
end

if angle ~= 0
    rad2deg = 360/(2*pi);
    g3 = imrotate(g3, -angle*rad2deg, 'bilinear', 'crop');
end

g3 = g3(r+1-out_radius:r+1+out_radius, r+1-out_radius:r+1+out_radius);
