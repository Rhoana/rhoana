function [borders adj conn npix dist valid avedist] = makeConn2(seg_vol, prob_vol)

sz = size(seg_vol);
segs = double(max(seg_vol(:)));
nmoves = 6;
moves = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];

borders = false(sz);

%Find border pixels
for mi = 1:nmoves
    direction = moves(mi,:);
    shifted = circshift(seg_vol, direction);
    
    %Blank out cube edges
    blank1 = 1:sz(1);
    blank2 = 1:sz(2);
    blank3 = 1:sz(3);
    if direction(1) == 1
        blank1 = 1;
    elseif direction(1) == -1
        blank1 = sz(1);
    end
    if direction(2) == 1
        blank2 = 1;
    elseif direction(2) == -1
        blank2 = sz(2);
    end
    if direction(3) == 1
        blank3 = 1;
    elseif direction(3) == -1
        blank3 = sz(3);
    end
    
    shifted(blank1, blank2, blank3) = 0;
    changed = shifted > 0 & seg_vol ~= shifted;
    borders(changed) = true;
end

borders = find(borders);

adj = zeros(length(borders),nmoves+1, 'int32');
adj(:,nmoves+1) = seg_vol(borders);

for mi = 1:nmoves
    direction = moves(mi,:);
    shifted = circshift(seg_vol, direction);
    
    %Blank out all the borders
    blank1 = 1:sz(1);
    blank2 = 1:sz(2);
    blank3 = 1:sz(3);
    if direction(1) == 1
        blank1 = 1;
    elseif direction(1) == -1
        blank1 = sz(1);
    end
    if direction(2) == 1
        blank2 = 1;
    elseif direction(2) == -1
        blank2 = sz(2);
    end
    if direction(3) == 1
        blank3 = 1;
    elseif direction(3) == -1
        blank3 = sz(3);
    end
    
    shifted(blank1, blank2, blank3) = 0;
    
    adj(:,mi) = shifted(borders);
    
end

fprintf(1, 'Making connectivity matrix (fast).\n');
%Now make a border connectivity network
conn = sparse([], [], [], segs, segs, segs*100);
%Also record the border surface area and distance / weight
npix = sparse([], [], [], segs, segs, segs*100);
dist = sparse([], [], [], segs, segs, segs*100);
avedist = sparse([], [], [], segs, segs, segs*100);

[sadj] = sort(adj(:,1:nmoves), 2, 'descend');
dadj = diff(sadj,1,2);
%Only join over 2-borders (ignore 3+ intersection points)
connadj = sum(dadj~=0,2)>=1;
compindex = find(connadj);
for pc = 1:length(compindex)
    px = compindex(pc);
    pairs = [1, find(dadj(px,:)) + 1];
    for pairix = 1:length(pairs)
        
        segi = adj(px,nmoves+1);
        segj = sadj(px,pairs(pairix));
        
        if segi ~= segj && segi ~= 0 && segj ~= 0
            conn(segi, segj) = 1;
            npix(segi, segj) = npix(segi, segj) + 1;
            %Distance calculation
            %dist(segi, segj) = dist(segi, segj) + prob_vol(borders(px)).^2;
            %TESTING
            dist(segi, segj) = full(dist(segi, segj)) + prob_vol(borders(px));
            %dist(segi, segj) = dist(segi, segj) + real(sqrt(prob_vol(borders(px))));
            %end            
        end

    end
    if (mod(pc, 500000) == 0)
        fprintf(1, 'Done %d of %d border pixels.\n', pc, length(compindex));
    end
end

%All connections are reciprocal
conn = max(conn, conn');
npix = npix + npix';
dist = dist + dist';

%Mean distance
valid = find(conn);
avedist(valid) = dist(valid)./npix(valid);

end
