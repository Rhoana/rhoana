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


function all1 = lines2allnn(lines1, xoffset, yoffset)
%Transform set of lines to a single struct

line0 = struct('x', {}, 'y', {}, 'angle', {}, 'junc', {}, 'lineix', {}, 'linepointix', {}, 'segment', {});

pointsn = 0;
for i = 1:length(lines1)
    pointsn = pointsn+length(lines1{i});
end
all1 = line0;
if pointsn > 0
    all1(pointsn).x = 0;
end

pointix = 0;
for i = 1:length(lines1)
    for j = 1:length(lines1{i})
        pointix = pointix + 1;
        all1(pointix).x = lines1{i}(j).segment.x + xoffset;
        all1(pointix).y = lines1{i}(j).segment.y + yoffset;
        all1(pointix).angle = lines1{i}(j).segment.angle;
        all1(pointix).junc = lines1{i}(j).segment.juncpart;
        all1(pointix).lineix = i;
        all1(pointix).linepointix = j;
        all1(pointix).segment = lines1{i}(j).segment;
    end
end

end