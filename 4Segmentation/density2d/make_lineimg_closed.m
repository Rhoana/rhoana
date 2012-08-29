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


function line_image = make_lineimg_closed (seg_list, ignore_segs, extra_lines, imgsize, maxspur, marginsize)
%Make a black and white (logical) image with white where lines occur
%ignore lines in the ignore_segs list and add lines in extra_lines

line_image = false(imgsize);
maxindex = double(max([seg_list.index]));
onbool = sparse([],[],[],1,maxindex,length(seg_list));
onbool([seg_list.index]) = 1;
onbool(ignore_segs) = 0;
onindex = find(onbool);
clear onbool;
segoff = seg_list(1).index - 1;
drawn = sparse([],[],[],maxindex,maxindex,3*length(seg_list));
for oni = onindex
    fromseg = seg_list(oni-segoff);
    if fromseg.linked_to ~= 0
        for toseg = fromseg.linked_to.values
            if any(onindex == toseg{1}.index) && ...
                    drawn(toseg{1}.index, fromseg.index) == 0
                %draw a line here
                xdiff = toseg{1}.x - fromseg.x;
                ydiff = toseg{1}.y - fromseg.y;
                npix = max(abs(xdiff), abs(ydiff));
                xl = fromseg.x:xdiff/npix:toseg{1}.x;
                yl = fromseg.y:ydiff/npix:toseg{1}.y;
                if xdiff == 0
                    xl = ones(size(yl))*fromseg.x;
                end
                if ydiff == 0
                    yl = ones(size(xl))*fromseg.y;
                end
                
                %Quick, non 4-connected
                %line_image(sub2ind(imgsize, round(yl), round(xl))) = true;
                
                %Slow, 4-connected
                %yr_prev = -1;
                %xr_prev = -1;
                for li = 1:length(xl)
                    yr = round(yl(li));
                    xr = round(xl(li));
                    line_image(yr, xr) = true;
                    %Ensure this is a 4-connected boundary
                    %if li > 1 && yr ~= yr_prev && xr ~= xr_prev
                    %    line_image(yr, xr_prev) = true;
                    %end
                    %yr_prev = yr;
                    %xr_prev = xr;
                    %imagesc(line_image);
                    %xlim([xr-10, xr+10]);
                    %ylim([yr-10, yr+10]);
                    %pause;
                end
                drawn(fromseg.index, toseg{1}.index) = 1;
            end
        end
    end
end

for exi = 1:length(extra_lines)
    eline = extra_lines{exi};
    for j = 1:length(eline)-1
        xfrom = eline(1,j);
        yfrom = eline(2,j);
        xto = eline(1,j+1);
        yto = eline(2,j+1);
        %draw a line here
        xdiff = xto - xfrom;
        ydiff = yto - yfrom;
        npix = max(abs(xdiff), abs(ydiff));
        xl = xfrom:xdiff/npix:xto;
        yl = yfrom:ydiff/npix:yto;
        if xdiff == 0
            xl = ones(size(yl))*xfrom;
        end
        if ydiff == 0
            yl = ones(size(xl))*yfrom;
        end
        
        %Quick, non 4-connected
        %line_image(sub2ind(imgsize, round(yl), round(xl))) = true;
        
        %Slow, 4-connected
        %yr_prev = -1;
        %xr_prev = -1;
        for li = 1:length(xl)
            yr = round(yl(li));
            xr = round(xl(li));
            line_image(yr, xr) = true;
            %Ensure this is a 4-connected boundary
            %if li > 1 && yr ~= yr_prev && xr ~= xr_prev
            %    line_image(yr, xr_prev) = true;
            %end
            %yr_prev = yr;
            %xr_prev = xr;
        end
    end
end

%Tidy up the image
ddisk4 = strel('disk', 4, 4);
%Fill any gaps (up to 4 px)
line_image = (imclose(line_image, ddisk4));
%Skeletonise
line_image = bwmorph(line_image, 'skel', Inf);

%Retain 4-connectedness
%line_image = bwmorph(line_image, 'diag', 1);
oneright = circshift(line_image, [1,0]);
oneright(1,:) = false;
ymove = circshift(line_image, [0,1]);
ymove(:,1) = false;
line_image((oneright & ymove)) = true;
ymove = circshift(line_image, [0,-1]);
ymove(:,size(ymove,2)) = false;
line_image(oneright & ymove) = true;

%Remove spurs of lenght maxspur or less
line_image = remove_spurs_margin(line_image, maxspur, marginsize);

%imagesc(line_image);
%pause(0.01);

end