function segments = clean1(segments)

    %Clean1
    sz = size(segments);
    nmoves = 26;
    moves = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1; ...
        1 1 0; -1 1 0; 1 -1 0; -1 -1 0; ...
        1 0 1; -1 0 1; 0 1 1; 0 -1 1; ...
        1 0 -1; -1 0 -1; 0 1 -1; 0 -1 -1; ...
        1 1 1; -1 1 1; 1 -1 1; -1 -1 1; ...
        1 1 -1; -1 1 -1; 1 -1 -1; -1 -1 -1];
    borders = int32(find(segments == 0));
    adj = zeros(length(borders),nmoves, 'int32');

    fprintf(1, 'Cleanup - calculating adjicency.\n');

    for mi = 1:nmoves
        direction = moves(mi,:);
        shifted = circshift(segments, direction);

        %Blank out all the borders
        blank1 = 1:sz(1);
        blank2 = 1:sz(2);
        blank3 = 1:sz(3);
        if direction(1) == 1
            blank1 = 1;
        elseif direction(1) == -1
            blank1 = sz(3);
        elseif direction(2) == 1
            blank2 = 1;
        elseif direction(2) == -1
            blank2 = sz(2);
        elseif direction(3) == 1
            blank3 = 1;
        elseif direction(3) == -1
            blank3 = sz(3);
        end

        shifted(blank1, blank2, blank3) = 0;

        adj(:,mi) = shifted(borders);

    end
    
    core = all(adj==0,2);
    adj(core) = -1;
    adjsame = adj(:,1);
    adjdiff = false(length(borders),1);
    
    for mi = 2:nmoves
        different = adj(:,mi) ~= 0 & adjsame ~= 0 & adjsame ~= adj(:,mi);
        adjdiff = adjdiff | different;
        new = adjsame == 0 & adj(:,mi) ~= 0;
        adjsame(new) = adj(new,mi);
    end
    
    cleanable = ~adjdiff & (adjsame > 0);
    
    %segtemp = segments;
    %segtemp(borders(core)) = 100;
    %segtemp(borders(cleanable)) = 200;
    %quickshowschsv(segtemp);
    segments(borders(cleanable)) = adjsame(cleanable);
    
end