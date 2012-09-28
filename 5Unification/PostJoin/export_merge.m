function export_merge (src_vol, seg_vol, alpha, dirname)
%Export 3D image as series of 2D images
%Labels: label images (16bit if possible otherwise 32bit)
%Overlay: seg_vol labels (random colors) over src_vol image (alpha blend)

labeldir = fullfile(dirname, 'labels');
overlaydir = fullfile(dirname, 'overlay');
mkdir(dirname);
mkdir(labeldir);
mkdir(overlaydir);

%compress seg_vol to as few labels as possible
fprintf(1, 'Compressing labels.\n');

[~, ~, new_vol] = unique(seg_vol);
new_vol = reshape(new_vol, size(seg_vol));

nsegments = max(new_vol(:));

segment_colours = hsv2rgb(...
    [rand(nsegments, 1), ...
    0.5 + rand(nsegments, 1) * 0.5, ...
    0.5 + rand(nsegments, 1) * 0.5]);

srcmi = min(src_vol(:));
srcma = max(src_vol(:));
if srcmi == srcma
    srcma = srcmi + 1;
end

fprintf(1, 'Saving files.\n');

for z = 1:min(size(src_vol,3),size(new_vol,3))
    
    filestring = sprintf('%4d', z-1);
    
    src_2d = src_vol(:,:,z);
    src_2d = repmat(src_2d, [1 1 3]);
    
    seg_2d = new_vol(:,:,z);
    %Write greyscale label image
    if nsegments <= 2^8-1
        imwrite(uint8(seg_2d), fullfile(labeldir, [filestring '.png']));
    elseif nsegments <= 2^16-1
        imwrite(uint16(seg_2d), fullfile(labeldir, [filestring '.png']));
    elseif nsegments <= 2^32-1
        fname = fullfile(labeldir, [filestring '.hdf5']);
        if ~exist(fname, 'file')
            h5create(fname, '/labels', size(seg_2d), 'DataType', 'uint32');
        end
        h5write(fname, '/labels', uint32(seg_2d));
    else
        fname = fullfile(labeldir, [filestring '.hdf5']);
        if ~exist(fname, 'file')
            h5create(fname, '/labels', size(seg_2d), 'DataType', 'uint64');
        end
        h5write(fullfile(labeldir, [filestring '.hdf5']), '/labels', uint64(seg_2d));
    end
    
    label_2d = uint8(ind2rgb(seg_2d, segment_colours) * 255);
    
    overlay = (1-alpha) * label_2d + alpha * src_2d;
  
    %Write overlay image
    imwrite(overlay, fullfile(overlaydir, [filestring '.png']));
      
end

    