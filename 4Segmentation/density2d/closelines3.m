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


function [mem_lines, segment_index] = closelines3(img, best_activ, best_ai, mem_lines, segment_index, zi, step, display, verbose)

if verbose
    disp('Closing edges.');
end

%Make an array of segments in index order
all_segs = segment_index.seg_index.values;
if isempty(all_segs)
    %Nothing to close!
    disp('Warning - empty trace.');
    return;
end
all_segs = [all_segs{:}];
asz = [all_segs.z];
all_segs = all_segs(asz==zi);
[segscore, segorder] = sort([all_segs(:).index]);
all_segs = all_segs(segorder);

%Check for (and remove) small loops (keep shortest path)
% unlinkedtot = 0;
% for segi = 1:length(all_segs)
%     linkseg = all_segs(segi);
%     if linkseg.linked_to ~= 0
%         linkto = linkseg.linked_to.values;
%         for l1 = 1:length(linkto)
%             link1 = linkto{l1};
%             link1reach = link1.linked_to.keys;
%             link1reach = [link1reach{:}];
%             for l2 = l1+1:length(linkto)
%                 link2 = linkto{l2};
%                 if any(link1reach==link2.index)
%                     %Unlink these
%                     d1 = sqrt((linkseg.x-link1.x)^2 + (linkseg.y-link1.y)^2);
%                     d2 = sqrt((linkseg.x-link2.x)^2 + (linkseg.y-link2.y)^2);
%                     d3 = sqrt((link2.x-link1.x)^2 + (link2.y-link1.y)^2);
%                     %Unlink the longest
%                     if d3 >= d1 && d3 >= d2
%                         %Unlink link1 and link2
%                         link1.linked_to.remove(link2.index);
%                         link2.linked_to.remove(link1.index);
%                     elseif d2 >= d1
%                         %Unlink linkseg and link2
%                         linkseg.linked_to.remove(link1.index);
%                         link1.linked_to.remove(linkseg.index);
%                     else
%                         %Unlink linkseg and link1
%                         linkseg.linked_to.remove(link2.index);
%                         link2.linked_to.remove(linkseg.index);
%                     end
%                     unlinkedtot = unlinkedtot + 1;
%                 end
%             end
%         end
%     end
% end
% fprintf(1,'Unlinked %d segments.\n', unlinkedtot);

%Close lines given by all_segs over lowest-pixel paths
imgsize = size(img);
margin = 35;
segsToClose = zeros(1,length(all_segs));
for segi = 1:length(all_segs)
    linkseg = all_segs(segi);
    if (linkseg.linked_to == 0 || linkseg.linked_to.length == 1) && ...
            linkseg.y > margin && linkseg.y < imgsize(1)-margin && ...
            linkseg.x > margin && linkseg.x < imgsize(2)-margin
        segsToClose(segi) = 1;
    end
end

%Smooth img
%gsd = 3;
%psf = fspecial('gaussian', gsd, gsd);
%img = imfilter(img, psf, 'symmetric', 'conv');

%Display the segs we need to close

%Make an image of lines
imlines = zeros(imgsize);
imclosepoints = zeros(imgsize);
imallpoints = zeros(imgsize);

rad = 5;
ddisk1 = strel('disk',rad, 4);
ddisk2 = strel('disk',rad+1, 4);

segoff = min([all_segs.index])-1;
nsegs = length(all_segs);
%Create a graph so we can calculate connection distances
cgraph = sparse([],[],[],nsegs,nsegs,nsegs*2);

for fsegi = 1:length(all_segs)
    fromseg = all_segs(fsegi);
    %Reverse index order (for dilation operation)
    fromix = fromseg.index - segoff;
    imallpoints(round(fromseg.y), round(fromseg.x)) = fromix;
%     spoint = zeros(imgsize);
%     spoint(round(fromseg.y), round(fromseg.x)) = 1;
%     spoint = imdilate(spoint, ddisk2);
%     biggerallpoints(spoint == 1) = fromseg.index;
    if segsToClose(fsegi) == 1
        imclosepoints(round(fromseg.y), round(fromseg.x)) = fromix;
    end
    if fromseg.linked_to == 0
        %Orphan
        imlines(fromseg.y, fromseg.x) = 1;
    else
        for toseg = fromseg.linked_to.values
            %Add a connection
            toix = toseg{1}.index - segoff;
            cgraph(fromix,toix) = 1;
            cgraph(toix,fromix) = 1;
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
            yr_prev = -1;
            xr_prev = -1;
            for li = 1:length(xl)
                yr = round(yl(li));
                xr = round(xl(li));
                imlines(yr, xr) = 1;
                %Ensure this is a 4-connected boundary
                if li > 1 && yr ~= yr_prev && xr ~= xr_prev
                    imlines(yr, xr_prev) = 1;
                end
                yr_prev = yr;
                xr_prev = xr;
            end
        end
    end
end

%Display lines
biglines = imdilate(imlines, ddisk1) > 0;
bigends = imdilate(imclosepoints>0, ddisk1) > 0;
biggerends = imdilate(imclosepoints, ddisk2) > 0;

%Docking points
minallpoints = imallpoints;
topap = max(imallpoints(:)) + 1;
minallpoints(imallpoints==0) = topap;
bigallpoints = imerode(minallpoints, ddisk1);
biggerallpoints = imerode(minallpoints, ddisk2);
dockpoints = biggerallpoints;
dockpoints(dockpoints==topap) = 0;
dockpoints(biglines) = 0;

%imagesc(biggerallpoints);
%pause();
%imagesc(dockpoints);
%pause();

if display
    closeimg = biglines;
    closeimg(bigends) = 0;
    closeimg(biggerends & ~bigends) = 1;
    colormap(gray);
    imagesc(closeimg);
    axis image off;
    pause(0.01);
end

imgscore = -best_activ;
maxscore = max(imgscore(:));
%imgscore(biglines) = max(imgscore(:));

%Calculate distances
if verbose
    disp('Calculating global distances.');
end
dgraph = all_shortest_paths(cgraph);

startrad = 40;
incrad = 5;
mindist = 5;
minmatch = 5*5;
toclose = find(segsToClose);
connimg = zeros(imgsize);

for tci = 1:length(toclose)
    
    searchrad = startrad;
    enough_points = false;
    while ~enough_points
        
        %Get the area around this point
        searchdisk = strel('disk',searchrad,4);
        closeseg = all_segs(toclose(tci));
        fy = closeseg.y;
        fx = closeseg.x;
        searchregion = zeros(imgsize);
        searchregion(fy,fx) = 1;
        searchregion = imdilate(searchregion, searchdisk)>0;

        %Find potential matches
        fromix = closeseg.index - segoff;
        reachable = find(dockpoints & searchregion);
        matchlist = [];
        for ri = 1:length(reachable)
            toix = dockpoints(reachable(ri));
            if fromix == toix
                %Same component - ignore
                %disp('Same segment.');
            elseif dgraph(fromix,toix) < mindist
                %disp('Already connected.');
            else
                %matchseg = all_segs(toix);
                %matchlist = [matchlist, sub2ind(imgsize, matchseg.y, matchseg.x)];
                matchlist = [matchlist, reachable(ri)];
            end
        end

        if length(matchlist) >= minmatch
            %fprintf(1,'Matching %d with %d possible dock points (rad=%d).\n', closeseg.index, length(matchlist), searchrad);
            enough_points = true;
        else
            %fprintf(1,'Only %d possible dock points at rad=%d. Growing search area.\n', length(matchlist), searchrad);
            searchrad = searchrad + incrad;
        end
    end

    %Search for the shortest path
    npoints = sum(searchregion(:));
    distgraph = sparse([],[],[],npoints,npoints,npoints*4);
    ppoints = find(searchregion);
    
    %Disallow travel over existing lines (allow from point)
    scoremask = biglines & (bigallpoints ~= closeseg.index - segoff);
    %imagesc(scoremask);
    %pause;
    searchscores = imgscore;
    searchscores(scoremask) = maxscore;
    %imagesc(searchscores);
    %pause(0.1);
    minval = double(min(searchscores(searchregion)))-1;
    for pix = 1:length(ppoints)
        indfrom = ppoints(pix);
        [ypix xpix] = ind2sub(imgsize,indfrom);
        for ymove = [0 1]
        %Note only +1 moves (reciprocal)
        yto = ypix + ymove;
        xto = xpix + (~ymove);
            if (yto > 0 && yto < imgsize(1) && ...
                xto > 0 && xto < imgsize(2) && ...
                searchregion(yto,xto))
                indto = sub2ind(imgsize, yto,xto);
                %distval = 10^double((img(indfrom)-minval) + (img(indto)-minval));
                %distval = double((img(indfrom)-minval) * (img(indto)-minval))^2;
                %distval = double(img(indfrom)) + double(img(indto));
                %distval = double(img(indfrom)-minval + img(indto)-minval + 1);
                %distval = double(img(indfrom)) * double(img(indto));
                
                %distval = double(searchscores(indfrom)) * double(searchscores(indto));
                %distval = double(searchscores(indfrom)-minval) * double(searchscores(indto)-minval);
                %distval = double(min(searchscores(indfrom), searchscores(indto))-minval);
                %distval = log10(double(min(searchscores(indfrom), searchscores(indto))-minval+1));
                distval = double(min(searchscores(indfrom), searchscores(indto))-minval)^2;
                %distval = 2^double(min(searchscores(indfrom), searchscores(indto))-minval);
                toix = find(ppoints == indto);
                distgraph(pix, toix) = distval;
                distgraph(toix, pix) = distval;
            end
        end
    end

    %Find the shortest path
    frompix = find(ppoints == sub2ind(imgsize, closeseg.y, closeseg.x));
    %disp('Calculating local distances.');
    [d pred] = shortest_paths(distgraph,frompix);
    shorti = 0;
    shortest = 0;
    bestmatchi = 0;
    for mi = 1:length(matchlist)
        matchix = find(ppoints == matchlist(mi));
        if isempty(matchix)
            disp('Warning - match expected but not found');
        elseif shorti == 0 || d(matchix) < shortest
            bestmatchi = mi;
            shorti = matchix;
            shortest = d(matchix);
        end
    end
    
    bestseg = all_segs(dockpoints(matchlist(bestmatchi)));
    if verbose
        fprintf(1,'Joining segments %d and %d.\n', closeseg.index, bestseg.index);
    end

    %Trace the line back
    currenti = shorti;
    manhattan_dist = 0;
    while currenti ~= frompix && currenti ~= 0
        [y x] = ind2sub(imgsize, ppoints(currenti));
        connimg(y,x) = 1;
        currenti = pred(currenti);
        manhattan_dist = manhattan_dist+1;
    end
    
    if manhattan_dist <= step
        %link directly
        closeseg.linkBoth(bestseg);
    else
        %break up the journey
        
        line0 = struct('segment', {}, 'next_angle', {}, 'next_angleindex', {}, 'next_direction', {}, 'stillgoing', {});
        thisline = line0;

        nstep = round(manhattan_dist/step);
        stepd = round(manhattan_dist/(nstep+1));
        nextstepi = stepd;
        
        stepi = 0;
        currenti = shorti;
        prevseg = bestseg;
        
        newi = length(thisline)+1;
        thisline(newi).segment = prevseg;
        thisline(newi).next_angle = prevseg.angle;
        thisline(newi).next_angleindex = prevseg.angleindex;
        thisline(newi).next_direction = 1;
        thisline(newi).stillgoing = true;
        
       while currenti ~= frompix && currenti ~= 0
            currenti = pred(currenti);
            stepi = stepi+1;
            if stepi == nextstepi
                [y x] = ind2sub(imgsize, ppoints(currenti));
                newlink = segment_index.getSegment(x,y,zi);
                if newlink == 0
                    newlink = segment_index.newSegment(x, y, zi, ...
                        prevseg.x-x, prevseg.y-y, 0, best_ai(y,x), best_activ(y,x), false);
                end
                
                if newlink.prev == 0
                    newlink.prev = prevseg;
                else
                    newlink.juncpart = true;
                end
                if prevseg.next == 0
                    prevseg.next = newlink;
                else
                    prevseg.juncpart = true;
                end
                prevseg.linkBoth(newlink);
                prevseg = newlink;
                nextstepi = nextstepi + stepd;
                
                newi = length(thisline)+1;
                thisline(newi).segment = prevseg;
                thisline(newi).next_angle = prevseg.angle;
                thisline(newi).next_angleindex = prevseg.angleindex;
                thisline(newi).next_direction = 1;
                thisline(newi).stillgoing = true;
                
                if nextstepi >= manhattan_dist
                    break;
                end
            end
        end
        
        if closeseg.prev == 0
            closeseg.prev = prevseg;
        else
            closeseg.juncpart = true;
        end
        if prevseg.next == 0
            prevseg.next = closeseg;
        else
            prevseg.juncpart = true;
        end
        closeseg.linkBoth(prevseg);
        
        newi = length(thisline)+1;
        thisline(newi).segment = closeseg;
        thisline(newi).next_angle = closeseg.angle;
        thisline(newi).next_angleindex = closeseg.angleindex;
        thisline(newi).next_direction = 1;
        thisline(newi).stillgoing = false;
       
        mem_lines{length(mem_lines)+1} = thisline;
        
    end
    
    if display
        imagesc((connimg>0) * 2 + (closeimg>0));
        axis image off;
        pause(0.01);
    end

end

%All end points are now connected
%figure();
% if display
%     colormap(gray);
%     result = imgscore;
%     result(closeimg>0) = max(result(:));
%     result(connimg>0) = min(result(:));
%     imagesc(result);
%     axis image off;
% end

end