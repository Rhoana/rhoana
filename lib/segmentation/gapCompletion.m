function segmentations = gapCompletion(imGray, imProb, threshRange, l_s_range, l_gc_range)

%threshRange = [0.3:0.01:0.4];
cs = 49;%49;%[19 29 49];
ms = 3; %3;
%l_s_range = [0:0.1:1]; %0;%[0:0.5:2];
%l_gc_range = [0:0.1:1];%2.0;

%fixed for probability map
mem = 1;%0;
m_std = 1.1;%1.5;%1.0;

imGray = norm01(adapthisteq(imGray));

counter = 0;
segmentations = zeros(size(imGray,1),size(imGray,2),...
    length(threshRange)*length(l_s_range)*length(l_gc_range));

[~, A_smooth, T_smooth, A_gc, T_gc] = ...
    getGraphCutConnectivityMatrizes2D(imGray, imProb, ...
    0.5, mem, m_std, cs, ms);

for i_thresh = 1:length(threshRange)
    T_prob = buildT([imProb(:)], threshRange(i_thresh));
    for i_l_s = 1:length(l_s_range)
        l_s = l_s_range(i_l_s);
        for i_l_gc = 1:length(l_gc_range)
            l_gc = l_gc_range(i_l_gc);
            
            [imSeg] = computeFlow2D_simple(imGray, T_prob, ...
                A_smooth*l_s, T_smooth*l_s, ...
                A_gc*l_gc, T_gc*l_gc);     
            
            imSeg = bwlabel(imSeg==0);
            imSeg = shrinkExtracellularSpace(imSeg);
            
            %delete regions too small to be a structure
            imSeg = filterSmallRegions(imSeg,30);
            imSeg = shrinkExtracellularSpace(imSeg);
            
            [~,~,mag] = gradientImg(imSeg,0);
            imSeg = ( mag>0 );
            
            %draw white boundary to have all membranes connected
            imSeg = drawFrame(imSeg,1,1)>0;
            
            %delete boundaries not part of the overall network
            imSeg = filterSmallRegions(imSeg,1000);
            
            counter=counter+1;
            segmentations(:,:,counter) = ~imSeg;
        end
    end
end
