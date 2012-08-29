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


function draw_patches(patches)

cumu_col = 0;

colormap(winter);
for i = 1:prod(size(patches))
    if ~isempty(patches{i})
        fill3(patches{i}{1}, patches{i}{2}, patches{i}{3}, ...
            (1:length(patches{i}{1})) + cumu_col,'EdgeColor','none');
        cumu_col = cumu_col + length(patches{i}{1});
        axis ij;
        axis image off;
        hold on;
    end
end
hold off;
view(2);
camlight;
lighting gouraud;
pause(2);

for vi = 1:5:10
    view(vi, 90-2*vi);
    pause(0.5);
end

end