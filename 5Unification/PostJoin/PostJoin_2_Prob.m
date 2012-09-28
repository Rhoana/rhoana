src_dir = 'C:\dev\datasets\conn\main_dataset\ac3train\';
dice_string = 'diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1';
result_name = 'res_from_sept_14_seg60_scf09_PF';
%result_name = 'res_from_sept_14_seg60_scf095_PF';
%result_name = 'res_from_sept_14_seg60_scf0975_PF';
nresult = 1;

if ~exist('seg_vol', 'var')
    fprintf(1, 'Loading working from PostJoin_Size.\n');
    load(sprintf('PostJoin_Size_working_%s.mat', result_name));
end

for minsegsize = [1000 2500 5000 10000 15000 20000]
    
    load(sprintf('PostJoin_%s_Size=%d.mat', result_name, minsegsize));
    
    seg_vol = segments;
    
    sz = size(seg_vol);
    segs = double(max(seg_vol(:)));
    
    fprintf(1, 'Calculating adjicency.\n');
    [borders adj conn npix dist valid avedist] = makeConn2(seg_vol, prob_vol);
    
    %Calculate segment sizes
    fprintf(1, 'Calculating segment sizes.\n');
    seg_stats = regionprops(seg_vol, 'Area', 'PixelIdxList');
    
    segsizes_i = [seg_stats.Area];
    
    %Now join up segments
    fprintf(1, 'Joining segments.\n');
    validsegs = segsizes_i > 0;
    
    for maxjoinscore = [0.1 0.2 0.3 0.4 0.5]
        %segoi = 1;
        fprintf(1,'Joining segments up to distance %1.2f.\n', maxjoinscore);
        %Blank out anything unconnected
        blankout = false(size(seg_vol));
        blanki = 0;
        
        becomes = zeros(1,segs);
        becomestest = 1:segs;
        joinphase = 0;
        
        brdRewrite = zeros(size(borders));
        
        while joinphase < 10 && min(avedist(conn>0)) < maxjoinscore
            
            joinphase = joinphase + 1;
            
            %find all close nodes
            invertdist = avedist;
            invert_index = find(invertdist);
            invertdist(invert_index) = max(invertdist(invert_index)) - invertdist(invert_index) + 1;
            isegdist_i = max(invertdist, [], 2);
            isegdist_i = full(isegdist_i(validsegs));
            [isegdist_o segorder] = sort(full(isegdist_i), 'descend');
            segv = find(validsegs);
            segorder = segv(segorder);
            
            joini = 0;
            for segi = segorder
                reach = find(conn(segi,:));
                if numel(reach) == 0
                    fprintf(1, 'WARNING: Segment %d has no neighbours.\n', segi);
                    continue
                end
                [mindist minr] = min(avedist(segi,reach));
                if validsegs(segi) && mindist <= maxjoinscore
                    
                    %Join this segment to its closest neighbour
                    neighbour = reach(minr);
                    %Sanity check
                    if mindist > maxjoinscore
                        fprintf(1, 'Warning - distance unexpectedly high. Joining anyway.');
                    end
                    becomes(segi) = neighbour;
                    becomes(becomes==segi) = neighbour;
                    if ~validsegs(neighbour)
                        neighbour = becomes(neighbour);
                        if neighbour == 0
                            fprintf(1, 'Warning - invalid neighbour.\n');
                            blankout(seg_vol == segi) = true;
                            blanki = blanki + 1;
                            validsegs(segi) = false;
                        else
                            becomes(segi) = neighbour;
                            becomes(becomes==segi) = neighbour;
                        end
                    end
                    if any(becomes==becomestest)
                        fprintf(1, 'Warning - loop created.\n');
                    end
                    
                    segsizes_i(neighbour) = segsizes_i(neighbour) + segsizes_i(segi);
                    
                    %Don't join the neighbour to itself
                    conn(segi,neighbour) = 0;
                    npix(segi,neighbour) = 0;
                    dist(segi,neighbour) = 0;
                    
                    %Join to other connections
                    update = find(conn(segi,:));
                    conn(neighbour,update) = 1;
                    npix(neighbour,update) = npix(neighbour,update) + npix(segi,update);
                    dist(neighbour,update) = dist(neighbour,update) + dist(segi,update);
                    avedist(neighbour,update) = dist(neighbour,update)./npix(neighbour,update);
                    
                    conn(update,neighbour) = conn(neighbour,update);
                    npix(update,neighbour) = npix(neighbour,update);
                    dist(update,neighbour) = dist(neighbour,update);
                    avedist(update,neighbour) = avedist(neighbour,update);
                    
                    %disconnect
                    %(no need to update disconnected distances)
                    conn(segi,update) = 0;
                    conn(update,segi) = 0;
                    conn(neighbour,segi) = 0;
                    conn(segi,neighbour) = 0;
                    
                    validsegs(segi) = false;
                    
                    %fprintf(1, 'dist = %1.6f : Joined segment %d (size = %1.2f) to %d (size = %1.2f).\n', ...
                    %    avedist(segi, neighbour), segi, segsizes_i(segi), neighbour, segsizes_i(neighbour));
                    
                    %write border to the output segments
                    %joinborder = (any(adj == segi,2) & any(adj == neighbour,2)) | brdRewrite == segi;
                    %brdRewrite(joinborder) = neighbour;
                    
                    %Modify the adjacency matrix
                    adj(adj==segi) = neighbour;
                    
                    joini = joini + 1;
                    
                    if mod(joini, 100) == 0
                        fprintf(1, 'Joined %d segments. Up to distance %d.\n', ...
                            joini, full(mindist));
                    end
                end
            end
            
            fprintf(1, 'Phase %d. Joined %d segments.\n', joinphase, joini);
            
        end
        
        %brdIndex = brdRewrite ~= 0;
        %segments(borders(brdIndex)) = brdRewrite(brdIndex);
        
        segments(blankout) = 0;
        fprintf(1, 'Blanked out %d unreachable islands.\n', blanki);
        
        rewrites = unique(becomes(becomes > 0));
        fprintf(1, 'Joining %d segments to %d supersegments.\n', joini, length(rewrites));
        
        if ~isempty(rewrites)
            for newseg = rewrites
                oldsegs = find(becomes==newseg);
                oldindex = false(sz);
                for oldsegi = oldsegs
                    oldindex(seg_vol == oldsegi) = true;
                end
                segments(oldindex) = newseg;
            end
        end
        
        %close all;
        %quickshowdualhsv_demo(start_img, segments(2:end-1,2:end-1,2:end-1))
        
        %disp('Cleaning...');
        %cleanseg = clean1(segments);
        %close all;
        %quickshowdualhsv_demo(start_img, cleanseg(2:end-1,2:end-1,2:end-1))
        
        %save(sprintf('PostJoin_%s_Size=%d.mat', result_name, minsegsize), 'segments', 'cleanseg');
        
        save(sprintf('PostJoin_%s_Size=%d_Prob=%1.2f.mat', result_name, minsegsize, maxjoinscore), 'segments');
        
        %prepare for more joining
        seg_vol = segments;
        
    end
end
