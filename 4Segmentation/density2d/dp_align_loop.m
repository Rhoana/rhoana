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


function [offsets, cumu_off patches] = dp_align_loop(lines1, lines2, zmatch, last_cumu_off, settings)
%Try to align images better based on line alignment

%Offset average threshold (pixels)
offset_thresh = 1;
trial_max = 20;

%Thickness of slice (in pixels - for display only)
z_size = 50;

Xpatches = zeros(4,1000);
Ypatches = zeros(4,1000);
Zpatches = zeros(4,1000);

zfrom = zmatch-1;

trials = 0;
stillgoing = 1;
xoffset = 0;
yoffset = 0;

[firstscore alignment alignmentP orph1P orph2P] = ...
    dp_align(lines1, lines2, xoffset, yoffset);
if isempty(alignment)
    offsets = [0; 0];
    cumu_off = last_cumu_off;
    return;
end
ave_offsetx = mean(alignment(1,:)-alignment(2,:));
ave_offsety = mean(alignment(3,:)-alignment(4,:));

lastscore = firstscore;
score = firstscore;

while stillgoing && trials <= trial_max
    
    if settings.verbose
        fprintf(1,'Alignment offset x%1.2f, y%1.2f, score=%1.2f (trial %d).\n', ave_offsetx, ave_offsety, score, trials);
    end
    
    stillgoing = 0;
    if abs(ave_offsetx) > offset_thresh
        %Move the lines to reduce the offset
        xoffset = xoffset + ave_offsetx;
        if settings.verbose
            fprintf(1,'Moved x offset by %1.2fpx.\n', ave_offsetx);
        end
        stillgoing = 1;
    end
    
    if abs(ave_offsety) > offset_thresh
        %Move the lines to reduce the offset
        yoffset = yoffset + ave_offsety;
        if settings.verbose
            fprintf(1,'Moved y offset by %1.2fpx.\n', ave_offsety);
        end
        stillgoing = 1;
    end
    
    trials = trials + 1;
    
    if ~stillgoing
        if settings.verbose
            disp('No action necessary.');
        end
        %pause;
    elseif trials <= trial_max
        %pause;
        [score next_alignment next_alignmentP next_orph1P next_orph2P] = ...
            dp_align(lines1, lines2, xoffset, yoffset);
        if score < lastscore
            %This is worse than the last point
            stillgoing = 0;
            score = lastscore;
            if settings.verbose
                disp('Score is worse - aborting.');
            end
        else
            lastscore = score;
            alignment = next_alignment;
            alignmentP = next_alignmentP;
            orph1P = next_orph1P;
            orph2P = next_orph2P;
            ave_offsetx = mean(alignment(1,:)-alignment(2,:));
            ave_offsety = mean(alignment(3,:)-alignment(4,:));
        end
    else
        if settings.verbose
            disp('Too many trials - abotring.');
        end
    end
    
end

if settings.verbose
    fprintf(1,'z2 to z%d, best offset found at x%1.2f, y%1.2f (%1.2f score gain).\n', zmatch, xoffset, yoffset, score-firstscore);
end

%Cumulative offset
offsets = [xoffset; yoffset];
cumu_off = last_cumu_off + [xoffset; yoffset];

patchx_offset = [last_cumu_off(1); cumu_off(1); cumu_off(1); last_cumu_off(1)];
patchy_offset = [last_cumu_off(2); cumu_off(2); cumu_off(2); last_cumu_off(2)];

%Match the segments
for matchi = 1:size(alignment,2)
    segA = alignmentP(1,matchi).segment;
    segB = alignmentP(2,matchi).segment;
    %Add to 3D
    segA.addZUpLink(segB);
    segB.addZDownLink(segA);
end

patches = {[],[],[]};

%%%% 3D view is slow can can cause compatibility issues on some
%%%% versions of Matlab - disabled. (edit below and density2d.m
%%%% to enable)
% if settings.display
% 
%     patch_num = 0;
% 
%     for matchi = 1:size(alignment,2)
%         %Try to find two surface patches for this alignment vertex
%         segA = alignmentP(1,matchi).segment;
%         segB = alignmentP(2,matchi).segment;
%         neighA = segA.linked_to;
%         neighB = segB.linked_to;
%         zmatchedA = [];
%         zmatchedB = [];
%         %Look for full surface (square)
%         if neighA ~= 0 && neighB ~= 0
%             zmatchedA = zeros(neighA.length);
%             zmatchedB = zeros(neighB.length);
%             neighsegA = neighA.values;
%             neighsegB = neighB.values;
%             for nextA = 1:neighA.length
%                 for nextB = 1:neighB.length
%                     segA2 = neighsegA{nextA};
%                     segB2 = neighsegB{nextB};
%                     if segA2.zlinked_up ~= 0 && segA2.zlinked_up.isKey(segB2.index)
%                         %Square match
%                         zmatchedA(nextA) = 1;
%                         zmatchedB(nextB) = 1;
%                         %These are aligned - add a square surface
%                         patch_num = patch_num + 1;
%                         Xpatches(:,patch_num) = [segA.x; segB.x; segB2.x; segA2.x] + patchx_offset;
%                         Ypatches(:,patch_num) = [segA.y; segB.y; segB2.y; segA2.y] + patchy_offset;
%                         Zpatches(:,patch_num) = [z_size*zfrom; z_size*zmatch; z_size*zmatch; z_size*zfrom];
%                     end
%                 end
%             end
%         end
% 
%         %Look for up triangles
%         if neighA ~= 0
%             neighsegA = neighA.values;
%             for nextA = 1:neighA.length
%                 if isempty(zmatchedA) || zmatchedA(nextA) == 0
%                     segA2 = neighsegA{nextA};
%                     %Unmatched A - add an up triangle surface
%                     patch_num = patch_num + 1;
%                     Xpatches(:,patch_num) = [segA.x; segB.x; segB.x; segA2.x] + patchx_offset;
%                     Ypatches(:,patch_num) = [segA.y; segB.y; segB.y; segA2.y] + patchy_offset;
%                     Zpatches(:,patch_num) = [z_size*zfrom; z_size*zmatch; z_size*zmatch; z_size*zfrom];
%                 end
%             end
%         end
% 
%         %Look for down triangles
%         if neighB ~= 0
%             neighsegB = neighB.values;
%             for nextB = 1:neighB.length
%                 if isempty(zmatchedB) || zmatchedB(nextB) == 0
%                     segB2 = neighsegB{nextB};
%                     %Unmatched B - add a down triangle surface
%                     patch_num = patch_num + 1;
%                     Xpatches(:,patch_num) = [segA.x; segB.x; segB2.x; segA.x] + patchx_offset;
%                     Ypatches(:,patch_num) = [segA.y; segB.y; segB2.y; segA.y] + patchy_offset;
%                     Zpatches(:,patch_num) = [z_size*zfrom; z_size*zmatch; z_size*zmatch; z_size*zfrom];
%                 end
%             end
%         end
% 
%         %All 3D patches have been added for this vertex
% 
%     end
% 
%     Xpatches = Xpatches(:,1:patch_num);
%     Ypatches = Ypatches(:,1:patch_num);
%     Zpatches = Zpatches(:,1:patch_num);
% 
% %    %3D patches complete - display
% %
% %     figure(2);
% %     set(gcf,'MenuBar', 'none');
% %     set(gcf,'ToolBar', 'none');
% %     scrsz = get(0,'ScreenSize');
% %     set(gcf,'Position',[30 50 scrsz(3)-90 scrsz(4)-90]);
% %     set(gca,'Position',[0.01 0.01 0.98 0.98])
% %     %fill3(Xpatches,Ypatches,Zpatches,'b','EdgeColor','none');
% %     fill3(Xpatches,Ypatches,Zpatches,1:length(Xpatches),'EdgeColor','none');
% %     colormap(winter);
% %     axis ij;
% %     axis image off;
% %     view(2);
% %     camlight;
% %     lighting gouraud;
% %     pause(0.1);
% %     % pause(2);
% %     % for vi = 1:5
% %     %     view(2*vi, 90-4*vi);
% %     %     pause(2);
% %     % end
% %     %pause;
%     
%     %Return variable if patch info is required
%     patches = {Xpatches, Ypatches, Zpatches};
%end

end
