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


function [score alignment alignment_points pointorph1_points pointorph2_points] = dp_align(lines1, lines2, xoffset, yoffset)

%Align the two sets of lines and return the cost based on:
%Point-to-point distance
euc_mult = 1;
%Angle difference
ang_mult = 20;
%Point consistency (one follows the other in a line)
gap_cost = 20;
%Minimum ends / junctions (possibly zero or very low?)
split_cost = 30;
%Cost orphans can be matched over
orph_match_max_cost = 10;

%Note that lines1 is being aligned to lines2.

%Ends free setting
endsfree = 1;

%Get single line versions
all1 = lines2allnn(lines1, 0, 0);
sizem = length(all1);

%Move all2 by the offset
all2 = lines2allnn(lines2, xoffset, yoffset);
sizen = length(all2);

% %Populate the cost matrix
%
% costs = zeros(sizem, sizen);
% 
% for i = 1:sizem
%     for j = 1:sizen
%         euc_cost = euc_mult*sqrt((all1(i).x-all2(j).x)^2 + (all1(i).y-all2(j).y)^2);
%         angle_diff = abs(all1(i).angle-all2(j).angle);
%         while angle_diff > pi/2
%             angle_diff = abs(angle_diff-pi);
%         end
%         angle_cost = ang_mult*angle_diff;
%         costs(i,j) = euc_cost + angle_cost;
%     end
% end

%Populate the cost matrix - matrix calculations for speed
%Calculate the euclidean distance cost
cost1 = (repmat([all1(:).x]',1,sizen) - repmat([all2(:).x], sizem, 1));
cost2 = (repmat([all1(:).y]',1,sizen) - repmat([all2(:).y], sizem, 1));
cost1 = euc_mult * sqrt( cost1.^2 + cost2.^2);

%Calculate the anglecost
cost2 = abs(repmat([all1(:).angle]', 1,sizen) - repmat([all2(:).angle], sizem, 1));
over = cost2 > pi/2;
while any(over(:))
    cost2(over) = abs(cost2(over)-pi);
    over = cost2 > pi/2;
end

%Cost matrix
costs = cost1 + ang_mult*cost2;
clear cost1 cost2 over;


costs = costs - gap_cost;

scores = zeros(sizem, sizen);
previ = zeros(sizem, sizen);
prevj = zeros(sizem, sizen);
ismatch = zeros(sizem, sizen);

split_cand = false(1,sizen);

for j = 1:sizen
    %Score filled in - check for junction or end of line
    if (j == 1 || j == sizen || all2(j).junc || ...
            all2(j).lineix ~= all2(j+1).lineix || ...
            all2(j).lineix ~= all2(j-1).lineix)
        %This is a junction or end of line candidate
        split_cand(j) = true;
    end
end

jsplit = find(split_cand);
notsplit = find(~split_cand);
upone = notsplit-1;
downone = notsplit+1;

%forwards
newjbool = [true, [all2(1:end-1).lineix] ~= [all2(2:end).lineix]];
samejix = find(~newjbool);
samejixupone = samejix-1;
newjix = find(newjbool);

%backwards
bnewjbool = [[all2(2:end).lineix] ~= [all2(1:end-1).lineix], true];
bsamejix = find(~bnewjbool);
bsamejixdownone = bsamejix+1;

for i = 1:sizem
    
%%%% Initialisation    
%%%% For Loop code - easier to understand, but runs slower
%    split_score = 0;
%    split_bestj = 0;
    
    if i == 1 || all1(i-1).lineix ~= all1(i).lineix
        if ~endsfree
%%%% For Loop code - easier to understand, but runs slower
%            %New line1 segment - all start with down moves
%            for j = 2:sizen
%                 if split_cand(j)
%                     %Junction or end - possible start point
%                     scores(i,j) = 0;
%                 else
%                     scores(i,j) = scores(i,j-1) - gap_cost;
%                     previ(i,j) = i;
%                     prevj(i,j) = j-1;
%                 end
%            end
            
%%%% Performs the above loop, but faster
            scores(i,split_cand) = 0;
            scores(i,notsplit) = scores(i,upone) - gap_cost;
            previ(i,notsplit) = i;
            prevj(i,notsplit) = upone;
            
%%%% For Loop code - easier to understand, but runs slower
%            for j = sizen-1:-1:1
%                 if ~split_cand(j)
%                     [scores(i,j) action] = max([scores(i,j), scores(i,j+1) - gap_cost]);
%                     if action == 2
%                         previ(i,j) = i;
%                         prevj(i,j) = j+1;
%                     end
%                 end
%            end

%%%% Performs the above loop, but faster
            [scores(i,notsplit) actions] = max([scores(i,notsplit), scores(i,downone) - gap_cost]);
            gapj = nosplit(actions==2);
            previ(i,gapj) = i;
            prevj(i,gapj) = gapj+1;
        end
        continue;
    end

%%%% PASS 1    
%%%% For Loop code - easier to understand, but runs slower
%     for j = samejix
%         %Forward matches
%         %Actions: 1 across -gap_cost, 2 down -gap_cost, 3 diag_down -cost
% 
%         across_score = scores(i-1,j) - gap_cost;
%         match_across = scores(i-1,j) - costs(i,j);
%         down_score = scores(i,j-1) - gap_cost;
%         %match_down = scores(i,j-1) - costs(i,j);
%         diagdn_score = scores(i-1,j-1) - costs(i,j);
%         %[best action] = max([across_score, match_across, down_score, match_down, diagdn_score]);
%         [best action] = max([across_score, match_across, down_score, diagdn_score]);
%         scores(i,j) = best;
%         switch(action)
%             case 1
%                 %gap across
%                 previ(i,j) = i-1;
%                 prevj(i,j) = j;
%             case 2
%                 %match across
%                 ismatch(i,j) = 1;
%                 previ(i,j) = i-1;
%                 prevj(i,j) = j;
%             case 3
%                 %gap down
%                 previ(i,j) = i;
%                 prevj(i,j) = j-1;
%             case 4
%                 % match diag down
%                 ismatch(i,j) = 1;
%                 ismatch(i-1,j-1) = 1;
%                 previ(i,j) = i-1;
%                 prevj(i,j) = j-1;
%         end
%     end
    
%%%% Performs the above loop, but faster

    scores(i,newjix) = scores(i-1,newjix) - gap_cost;
    previ(i,newjix) = i-1;
    prevj(i,newjix) = newjix;

    prevscores = scores(i,:);
    checkjix = samejix;
    checkjixupone = samejixupone;
    
    across_score = scores(i-1,checkjix) - gap_cost;
    match_across = scores(i-1,checkjix) - costs(i,checkjix);
    diagdn_score = scores(i-1,checkjixupone) - costs(i,checkjix);
    [best action] = max([across_score; match_across; diagdn_score]);
    scores(i,checkjix) = best;

    %1 gap across
    aj = checkjix(action==1);
    previ(i,aj) = i-1;
    prevj(i,aj) = aj;

    %2 match across
    aj = checkjix(action==2);
    ismatch(i,aj) = 1;
    previ(i,aj) = i-1;
    prevj(i,aj) = aj;

    %3 match diag down
    aj = checkjix(action==3);
    diagmod = zeros(1,sizen);
    diagmod(checkjix(action==3)) = 1;
    diagmod(ismatch(i-1,:)==1) = 0;
    ismatch(i,aj) = 1;
    ismatch(i-1,aj-1) = 1;
    previ(i,aj) = i-1;
    prevj(i,aj) = aj-1;

    %Check if any have changed
    checkjix = find(prevscores ~= scores(i,:)) + 1;
    if ~isempty(checkjix) && checkjix(end) == sizen+1
        checkjix = checkjix(1:end-1);
    end
    checkjixupone = checkjix-1;
    prevscores = scores(i,:);
    
    ntimes = 1;
    
    %Do the gap down scores in a loop
    while any(checkjix)
        %Recalc the gap-down moves for any better scores
        down_score = scores(i,checkjixupone) - gap_cost;
        %match_down = scores(i,checkjixupone) - costs(i,checkjix);
        [best action] = max([scores(i,checkjix); down_score]);
        scores(i,checkjix) = best;

        %(1 is the same score)

        %2 gap down
        aj = checkjix(action==2);
        previ(i,aj) = i;
        prevj(i,aj) = aj-1;
        %Tidy up previous scores
        ismatch(i,aj) = 0;
        diagcheck = zeros(1,sizen);
        diagcheck(checkjix(action==2)) = 1;
        diagcheck = find(diagcheck & diagmod);
        ismatch(i-1,diagcheck-1) = 0;

        %Check if any have changed
        checkjix = find(prevscores ~= scores(i,:)) + 1;
        if ~isempty(checkjix) && checkjix(end) == sizen+1
            checkjix = checkjix(1:end-1);
        end
        checkjixupone = checkjix-1;
        prevscores = scores(i,:);
        
        ntimes = ntimes + 1;
    end
    

%%%% PASS 2
%%%% For Loop code - easier to understand, but runs slower
%      for j = 1:sizen
% %         %Backward matches
% %         %Actions: 1 same, 2 up -gap_cost, 3 diag_up -cost
%         if j == sizen || all2(j+1).lineix ~= all2(j).lineix
%             %New line2 segment can't move up so leave the same.
%             %(already filled in)
%         else
%             up_score = scores(i,j+1) - gap_cost;
%             %Note there is no match up - this causes loops in the matrix
%             %Orphan cleanup should catch these instead
%             %match_up = scores(i,j+1) - costs(i,j);
%             diagup_score = scores(i-1,j+1) - costs(i,j);
%             [best action] = max([scores(i,j), up_score, diagup_score]);
%             switch(action)
%                 case 1
%                     %same
%                 case 2
%                     %gap up
%                     scores(i,j) = best;
%                     ismatch(i,j) = 0;
%                     previ(i,j) = i;
%                     prevj(i,j) = j+1;
%                 case 3
%                     %match diag up
%                     scores(i,j) = best;
%                     ismatch(i,j) = 1;
%                     ismatch(i-1,j+1) = 1;
%                     previ(i,j) = i-1;
%                     prevj(i,j) = j+1;
%             end
%         end
%         
%         %Scores are filled in - check for the best split point (greedy)
%         if split_cand(j) && (split_score == 0 || split_score < scores(i,j))
%             split_score = scores(i,j);
%             split_bestj = j;
%         end
%         
%      end

%%%% Performs the above loop, but faster

    %Backwards matches
    up_score = scores(i,bsamejixdownone) - gap_cost;
    diagup_score = scores(i-1,bsamejixdownone) - costs(i,bsamejix);
    [best action] = max([scores(i,bsamejix); up_score; diagup_score]);
    
    %(1 is the same score)
    %2 gap up
    aj = bsamejix(action==2);
    scores(i,aj) = best(action==2);
    ismatch(i,aj) = 0;
    previ(i,aj) = i;
    prevj(i,aj) = aj+1;
    
    %3 match diag up
    aj = bsamejix(action==3);
    scores(i,aj) = best(action==3);
    ismatch(i,aj) = 1;
    ismatch(i-1,aj+1) = 1;
    previ(i,aj) = i-1;
    prevj(i,aj) = aj+1;

    
%%%% PASS 3

    %Scores are filled in - check for the best split point (greedy)
    [split_score candix] = max(scores(i,jsplit));
    split_bestj = jsplit(candix);
    
    %Allow jumps at split points
    split_score = split_score - split_cost;
    for j = find(split_cand)
        if j ~= split_bestj && scores(i,j) < split_score
            scores(i,j) = split_score;
            previ(i,j) = i;
            prevj(i,j) = split_bestj;
        end
    end
    
end

score = 0;
matches = zeros(1,sizem);
matchesji = zeros(1,sizem);
orph1 = ones(1,sizem);
orph2 = ones(1,sizen);

path = zeros(sizem, sizen);

displaymatches ([], [[all1.x]; [all1.y]], [[all2.x]; [all2.y]]);
%pause;
pause(0.1);

%Trace back the matches
for i = sizem:-1:1
    if i == sizem || (all1(i).lineix ~= all1(i+1).lineix)
        %This is the end of a line - trace the match back
        mi = i;
        [bestscore mj] = max(scores(i,:));
        score = score + bestscore;
        while ~isempty(mj) && mi~=0 && mj~=0
            if ismatch(mi, mj)
                matches(mi) = mj;
                orph1(mi) = 0;
                orph2(mj) = 0;
            end
            path(mi,mj) = 1;
            minew = previ(mi,mj);
            mj = prevj(mi,mj);
            mi = minew;
        end
    end
end

%Display alignment state
orphi = find(orph1);
pointorph1 = [[all1(orphi).x]; [all1(orphi).y]];
orphi = find(orph2);
pointorph2 = [[all2(orphi).x]; [all2(orphi).y]];

match_fromi = find(matches>0);
match_fromj = find(matchesji>0);
matchi = [match_fromi, matchesji(match_fromj)];
matchj = [matches(match_fromi), match_fromj];
pointmatches = [[all1(matchi).x]; [all2(matchj).x]; [all1(matchi).y]; [all2(matchj).y]];

displaymatches (pointmatches, pointorph1, pointorph2);
%pause;
pause(0.1);

%%%% Optional extra pass to extend alignment to neighbouring segments
%Match any unmatched i to the prev or next or their neighbours
omatched = 0;
for i = 1:sizem
    if orph1(i)
        %Find candidates that match neighbours (in other section)
        matchj = zeros(1,16);
        maxmj = 1;
        if i > 1 && (all1(i).lineix == all1(i-1).lineix)
            newmatch = find(ismatch(i-1,:));
            matchj(maxmj:maxmj+length(newmatch)-1) = newmatch;
            maxmj = maxmj + length(newmatch);
        end
        if i < sizem && (all1(i).lineix == all1(i+1).lineix)
            newmatch = find(ismatch(i+1,:));
            matchj(maxmj:maxmj+length(newmatch)-1) = newmatch;
            maxmj = maxmj + length(newmatch);
        end
        %Also consider neighbours of these candidates
        jlen = maxmj-1;
        for jix = 1:jlen
            if matchj(jix) > 1 && (all2(matchj(jix)).lineix == all2(matchj(jix)-1).lineix)
                matchj(maxmj) = matchj(jix)-1;
                maxmj = maxmj + 1;
            end
            if matchj(jix) < sizen && (all2(matchj(jix)).lineix == all2(matchj(jix)+1).lineix)
                matchj(maxmj) = matchj(jix)+1;
                maxmj = maxmj + 1;
            end
        end
        %Find the best candidate
        matchj = matchj(1:maxmj-1);
        mcost = costs(i,matchj);
        [mc mi] = min(mcost);
        bestj = matchj(mi);
        if mc <= orph_match_max_cost
            matches(i) = bestj;
            orph1(i) = 0;
            orph2(bestj) = 0;
            omatched = omatched + 1;
        end
    end
end
%fprintf(1,'Adopted %d i orphan(s).\n', omatched);

%Match any unmatched j to the prev or next or their neighbours
omatched = 0;
for j = 1:sizen
    if orph2(j)
        %Find candidates that match neighbours (in other section)
        matchi = zeros(1,16);
        maxmi = 1;
        if j > 1 && (all2(j).lineix == all2(j-1).lineix)
            newmatch = find(ismatch(:,j-1))';
            matchi(maxmi:maxmi+length(newmatch)-1) = newmatch;
            maxmi = maxmi + length(newmatch);
        end
        if j < sizen && (all2(j).lineix == all2(j+1).lineix)
            newmatch = find(ismatch(:,j+1))';
            matchi(maxmi:maxmi+length(newmatch)-1) = newmatch;
            maxmi = maxmi + length(newmatch);
        end
        %Also consider neighbours of these candidates
        ilen = maxmi-1;
        for iix = 1:ilen
            if matchi(iix) > 1 && (all1(matchi(iix)).lineix == all1(matchi(iix)-1).lineix)
                matchi(maxmi) = matchi(iix)-1;
                maxmi = maxmi + 1;
            end
            if matchi(iix) < sizem && (all1(matchi(iix)).lineix == all1(matchi(iix)+1).lineix)
                matchi(maxmi) = matchi(iix)+1;
                maxmi = maxmi + 1;
            end
        end
        %Find the best candidate
        matchi = matchi(1:maxmi-1);
        mcost = costs(matchi,j);
        [mc mi] = min(mcost);
        besti = matchi(mi);
        if mc <= orph_match_max_cost
            matchesji(j) = besti;
            orph1(besti) = 0;
            orph2(j) = 0;
            omatched = omatched + 1;
        end
    end
end
%fprintf(1,'Adopted %d j orphan(s).\n', omatched);

%Record x y points for the matches and orphans

orphi = find(orph1);
pointorph1 = [[all1(orphi).x]; [all1(orphi).y]];
pointorph1_points = all1(orphi);
orphi = find(orph2);
pointorph2 = [[all2(orphi).x]; [all2(orphi).y]];
pointorph2_points = all2(orphi);

match_fromi = find(matches>0);
match_fromj = find(matchesji>0);
matchi = [match_fromi, matchesji(match_fromj)];
matchj = [matches(match_fromi), match_fromj];
pointmatches = [[all1(matchi).x]; [all2(matchj).x]; [all1(matchi).y]; [all2(matchj).y]];

alignment = pointmatches;
alignment_points = [all1(matchi); all2(matchj)];

displaymatches (pointmatches, pointorph1, pointorph2);
%pause;
pause(0.1);

resultalign = zeros(sizem, sizen);
for i = 1:sizem
    if matches(i) > 0
        resultalign(i,matches(i)) = 1;
    end
end

end

function displaymatches (pointmatches, pointorph1, pointorph2)
countm = size(pointmatches,2);
labels = {};
subplot(2,3,[2 3 5 6]);
set(gca,'Position',[0.34 0.01 0.65 0.99])
if ~isempty(pointorph1)
    plot(pointorph1(1,:), pointorph1(2,:), 'xk');
    hold on;
    labels{length(labels)+1} = 'z=1';
end
if ~isempty(pointorph2)
    plot(pointorph2(1,:), pointorph2(2,:), 'ok');
    hold on;
    labels{length(labels)+1} = 'z=2';
end
if countm > 0
    labels{length(labels)+1} = 'aligned';
end
for m = 1:countm
    plot(pointmatches(1:2,m), pointmatches(3:4,m), '.-k');
    %hold on;
end
axis ij image off;
legend(labels,'Location','SouthEast');
hold off;
end