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


classdef LineSegment < handle
    %LINESEGMENT Represents a single segment of line
    %Contains location, angle and score information
    %Links to other segments (many to many)
    %Maintains next / previous order
    
    properties
        index = int32(0);
        x = 0;
        y = 0;
        z = 0;
        xoffset = 0;
        yoffset = 0;
        angle = 0;
        angleindex = 0;
        %next_direction = 0;
        score = 0;
        juncpart = false;
        type_syn = false;
        type_mito = false;
        linked_to = 0;
        prev = 0;
        next = 0;
        zlinked_up = 0;
        zlinked_down = 0;
        visited = false;
    end
    
    methods
        
        %Constructor
        function ls = LineSegment(index, x, y, z, xo, yo, angle, angleindex, score, juncpart)
            if nargin > 0 % Support calling with 0 arguments
                ls.index = int32(index);
                ls.x = x;
                ls.y = y;
                ls.z = z;
                ls.xoffset = xo;
                ls.yoffset = yo;
                ls.angle = angle;
                ls.angleindex = angleindex;
                %ls.next_direction = next_direction;
                ls.score = score;
                ls.juncpart = juncpart;
                ls.type_syn = false;
                ls.type_mito = false;
                ls.linked_to = 0;
                ls.next = 0;
                ls.prev = 0;
                ls.zlinked_up = 0;
                ls.zlinked_down = 0;
                ls.visited = 0;
            end
        end
        
        function ls2 = clone(ls)
            %Clone returns a new LineSegment with the same properties and
            %location as the original, but without any index or links
            %prev and next are preserved, but not reciprocal.
            ls2 = LineSegment();
            ls2.index = 0;
            ls2.x = ls.x;
            ls2.y = ls.y;
            ls2.z = ls.z;
            ls.xoffset = ls.xoffset;
            ls.yoffset = ls.yoffset;
            ls.angle = ls.angle;
            ls.angleindex = ls.angleindex;
            %ls.next_direction = ls.next_direction;
            ls2.score = ls.score;
            ls2.juncpart = ls.juncpart;
            ls2.type_syn = ls.type_syn;
            ls2.type_mito = ls.type_mito;
            ls2.linked_to = 0;
            ls2.next = ls.next;
            ls2.prev = ls.prev;
            ls.zlinked_up = 0;
            ls.zlinked_down = 0;
            ls.visited = 0;
        end
        
        function setNext(ls, nextls)
            ls.next = nextls;
            nextls.prev = ls;
            ls.linkBoth(nextls);
        end
        
        function setPrev(ls, prevls)
            ls.prev = prevls;
            prevls.next = ls;
            ls.linkBoth(nextls);
        end
        
        function c = linkCount(ls)
            if ls.linked_to == 0
                c = 0;
            else
                c = ls.linked_to.length;
            end
        end
        
        %One way link
        function addLink(ls, child)
            if ls.linked_to == 0
                ls.linked_to = containers.Map(child.index, child);
            else
                ls.linked_to(child.index) = child;
            end
        end
        
        %One way link
        function addZUpLink(ls, child)
            if ls.zlinked_up == 0
                ls.zlinked_up = containers.Map(child.index, child);
            else
                ls.zlinked_up(child.index) = child;
            end
        end
        
        %One way link
        function addZDownLink(ls, child)
            if ls.zlinked_down == 0
                ls.zlinked_down = containers.Map(child.index, child);
            else
                ls.zlinked_down(child.index) = child;
            end
        end
        
        %Join up segments - two way link
        function linkBoth (ls, next_segment)
            if ls.z == next_segment.z
                ls.addLink(next_segment);
                next_segment.addLink(ls);
            elseif ls.z > next_segment.z
                ls.addZDownLink(next_segment);
                next_segment.addZUpLink(ls);
            else
                ls.addZUpLink(next_segment);
                next_segment.addZDownLink(ls);
            end
        end
        
        function disconnectAll(ls)
            linked_keys = keys(ls.linked_to);
            for k = linked_keys
                joined = ls.linked_to(k{1});
                joined.linked_to.remove(ls.index);
                ls.linked_to.remove(joined.index);
            end
            linked_keys = keys(ls.zlinked_up);
            for k = linked_keys
                joined = ls.zlinked_up(k{1});
                joined.zlinked_down.remove(ls.index);
                ls.zlinked_up.remove(joined.index);
            end
            linked_keys = keys(ls.zlinked_down);
            for k = linked_keys
                joined = ls.zlinked_down(k{1});
                joined.zlinked_up.remove(ls.index);
                ls.zlinked_down.remove(joined.index);
            end
        end
        
        function visit(ls)
            ls.visited = true;
        end
        
        function reset(ls)
            ls.visited = false;
        end
        
    end
    
end

