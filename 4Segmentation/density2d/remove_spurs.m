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


function start_img = remove_spurs(start_img, maxspurdist)

imgsize = size(start_img);
safe = maxspurdist+1;

npix = sum(start_img(:));
cgraph = sparse([],[],[],npix,npix,npix*4);
pix_index = find(start_img);

%spurs = false(imgsize);

for pix = 1:length(pix_index)
    indfrom = pix_index(pix);
    [ypix xpix] = ind2sub(imgsize,indfrom);
    for ymove = [0 1]
        %Note only +1 moves (reciprocal)
        yto = ypix + ymove;
        xto = xpix + (~ymove);
        if (yto > 0 && yto < imgsize(1) && ...
                xto > 0 && xto < imgsize(2) && ...
                start_img(yto,xto))
            indto = sub2ind(imgsize, yto,xto);
            toix = find(pix_index == indto);
            cgraph(pix, toix) = 1;
            cgraph(toix, pix) = 1;
        end
    end
end

nconn = sum(cgraph);
deadends = find(nconn == 1);
junctions = find(nconn > 2);
juncorend = find((nconn > 2) | (nconn == 1));

%Display working
% ddisk4 = strel('disk', 4, 4);
% disp_img = zeros(imgsize, 'uint8');
% disp_img(pix_index(juncorend)) = 170;
% disp_img(pix_index(deadends)) = 85;
% disp_img = imdilate(disp_img, ddisk4);
% disp_img(start_img) = 255;
% colormap(hot);
% imagesc(disp_img);
% pause;

for deix = 1:length(deadends)
    [y x] = ind2sub(imgsize,pix_index(deadends(deix)));
    if start_img(y,x)
        %Spur has not been removed yet
        [d dt pred] = bfs(cgraph,deadends(deix));
        dist = d(juncorend);
        dist(dist==-1) = safe;
        dist(dist==0) = safe;
        
        [mindist ji] = min(dist);
        
        if mindist <= maxspurdist
            
            %fprintf(1,'Found a spur, length %d.\n', mindist);
            
            %Display working
%             imagesc(start_img);
%             hold on;
%             %[y x] = ind2sub(imgsize,pix_index(deadends(deix)));
%             plot(x,y,'bo');
%             hold off;
%             %pause;
            
            %This is a spur - remove it from the start_img image
            pix = juncorend(ji);
            %Dont remove junction points
            while any(junctions == pix) && pix ~= 0 && pix ~= -1
                pix = pred(pix);
            end
            %endix = deadends(deix);
            while pix ~= 0 && pix ~= -1
                indfrom = pix_index(pix);
                [ypix xpix] = ind2sub(imgsize,indfrom);
                start_img(ypix, xpix) = false;
                %spurs(ypix, xpix) = true;
                pix = pred(pix);
            end
            
        end
    end
    
end
end