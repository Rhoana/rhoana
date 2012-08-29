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


classdef SegmentIndex < handle
    %SEGMENTINDEX Contains all segments indexed by location
    %Segments are instances of LineSegment
    
    properties
        seg_index = containers.Map;
        seg_count = 0;
    end
    
    methods
        
        %Constructor
        function si = SegmentIndex()
            si.seg_index = containers.Map;
            si.seg_count = 0;
        end
        
        function ls = newSegment(si, x, y, z, xo, yo, angle, angleindex, score, juncpart)
            si.seg_count = si.seg_count + 1;
            ls = LineSegment(si.seg_count, x, y, z, xo, yo, angle, angleindex, score, juncpart);
            location = sprintf('%d:%d:%d', x, y, z);
            si.seg_index(location) = ls;
        end
        
        function ls = newSegmentP(si, point)
            ls = si.newSegment(point.x, point.y, point.z, point.xo, point.yo, ...
                point.angle, point.angleindex, point.score, point.juncpart);
        end
        
        function ls = newSegmentL(si, ls)
            si.seg_count = si.seg_count + 1;
            ls.index = si.seg_count;
            location = sprintf('%d:%d:%d', ls.x, ls.y, ls.z);
            si.seg_index(location) = ls;
        end
        
        function seg = getSegment(si, x, y, z)
            location = sprintf('%d:%d:%d', x, y, z);
            if si.seg_index.isKey(location)
                seg = si.seg_index(location);
            else
                seg = 0;
            end
        end
        
        function seg = getSegmentP(si, point)
            seg = si.getSegment(point.x, point.y, point.z);
        end
        
        function seg = getSegmentL(si, ls)
            seg = si.getSegment(ls.x, ls.y, ls.z);
        end
    end
    
end

