src_dir = 'C:\dev\datasets\conn\main_dataset\ac3train\';
dice_string = 'diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1';
result_name = 'res_from_sept_14_seg60_scf095_PF';
%result_name = 'res_from_sept_14_seg60_scf0975_PF';
nresult = 1;

for minsegsize = 10000%[1000 2500 5000 10000]
    load(sprintf('PostJoin_%s_Size=%d.mat', result_name, minsegsize));
    
    %downsample the volume for rendering
    sz = size(segments);
    xmax = floor(sz(1)/4);
    ymax = floor(sz(2)/4);
    zmax = floor(sz(3)/2);
    seg_scale = zeros(xmax, ymax, zmax, 'int32');
    for imgi = 1:zmax
        seg_scale(:,:,imgi) = imresize(segments(:,:,imgi*2), [xmax ymax], 'nearest');
    end
    clearvars segments;
    maxseg = max(seg_scale(:));
    segcounts = hist(single(seg_scale(:)),single(0:maxseg));
    %0 is the boundary
    segcounts = segcounts(2:end);
    [segc_sorted segcountorder] = sort(segcounts, 'descend');
    maxview = 60;
    allf = cell(1,maxview);
    allv = cell(1,maxview);
    allc = cell(1,maxview);
    
    figure(1);
    set(gcf,'Renderer','OpenGL');
    
    for view_i = 1:min(length(segcountorder),maxview)
        view_segi = segcountorder(view_i);
        shape = seg_scale == view_segi;
        shapesize = sum(shape(:));
        fprintf(1,'Got a shape, size %d (expected %d).\n', shapesize, segc_sorted(view_i))
        
        [f v] = isosurface(shape,0.5);
        allf{view_i} = f;
        allv{view_i} = v;
        allc{view_i} = repmat(view_i, size(v,1), 1);
        
        clf;
        %for patch_i = [1 view_i]
        for patch_i = [9 view_i]
            p = patch('Faces', allf{patch_i}, 'Vertices', allv{patch_i}, 'FaceVertexCData', allc{patch_i}, 'FaceColor', 'interp', 'edgecolor', 'none');
            hold on;
        end
        hold off;
        colormap(jet);
        
        view(3);
        axis tight;
        camlight
        lighting gouraud
        title(num2str(view_i));
        pause(0.5);
        
    end
    
    clf;
    goodpatch = [9:30];
    cols = jet(length(goodpatch));
    for pai = 1:length(goodpatch)
        patch_i = goodpatch(pai);
        p = patch('Faces', allf{patch_i}, 'Vertices', allv{patch_i}, 'FaceVertexCData', allc{patch_i}, 'FaceColor', cols(pai,:), 'edgecolor', 'none');
        hold on;
    end
    hold off;
    colormap([hsv; jet]);
    view(10,20);
    axis tight;
    
    view(3);
    axis tight;
    camlight
    lighting gouraud
    pause();
    
end