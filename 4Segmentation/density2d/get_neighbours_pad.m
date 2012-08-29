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


function hood = get_neighbours_pad(img, x, y, rad_xy)
    
    w_xy = rad_xy*2+1;
    hood = zeros(w_xy, w_xy, class(img));
    
    xi = max(x-rad_xy,1):min(x+rad_xy,size(img,1));
    yi = max(y-rad_xy,1):min(y+rad_xy,size(img,2));
    
    xh = xi - x + rad_xy + 1;
    yh = yi - y + rad_xy + 1;
    
    hood(xh, yh) = img(xi,yi);
    
end