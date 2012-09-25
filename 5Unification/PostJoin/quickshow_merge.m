function quickshow_merge (src_vol, seg_vol, alpha)
%Display a 3D image as a movie of 2D images
%Overlay seg_vol labels over src_vol image

nsegments = max(seg_vol(:));

segment_colours = hsv2rgb(...
    [rand(nsegments, 1), ...
    0.5 + rand(nsegments, 1) * 0.5, ...
    0.5 + rand(nsegments, 1) * 0.5]);

srcmi = min(src_vol(:));
srcma = max(src_vol(:));
if srcmi == srcma
    srcma = srcmi + 1;
end

% scrsz = get(0,'ScreenSize');
figure(1);
% set(gcf,'MenuBar', 'none');
% set(gcf,'ToolBar', 'none');
% set(1,'Position',[38,70, scrsz(3)/2-48, scrsz(4)-148]);
% set(gca,'Position',[0 0 1 1]);
% figure(2);
% set(gcf,'MenuBar', 'none');
% set(gcf,'ToolBar', 'none');
% set(2,'Position',[scrsz(3)/2,70, scrsz(3)/2-48, scrsz(4)-148]);
% set(gca,'Position',[0 0 1 1]);

for z = 1:min(size(src_vol,3),size(seg_vol,3))
    
    %src_2d = uint8(src_vol(:,:,z) / srcma * 255);
    src_2d = src_vol(:,:,z);
    src_2d = repmat(src_2d, [1 1 3]);
    
    seg_2d = seg_vol(:,:,z);
    label_2d = uint8(ind2rgb(seg_2d, segment_colours) * 255);
    
    overlay = (1-alpha) * label_2d + alpha * src_2d;
  
    imshow(overlay);
    axis image off;
    
    pause(0.2);
    if z == 1
        pause(1);
    end
    
end

    