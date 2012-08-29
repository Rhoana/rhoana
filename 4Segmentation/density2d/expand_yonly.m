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


function images_exp = expand_yonly(images_r, w)
initsize = size(images_r,2);
images_r(1,initsize*2) = 0;

%Mirror images (in y direction only) to eliminate any bias.
for i = 1:initsize
    %Flip y
    flip = reshape(images_r(:,i), w, w);
    flip = flip(w:-1:1,:);
    images_r(:,i+initsize) = flip(:);
end

images_exp = images_r;

end