function [labelfile new_vol] = export_merge_omni (src_vol, seg_vol, dirname)
%Export 3D image as series of 2D images
%Labels: label images (16bit if possible otherwise 32bit)
%Overlay: seg_vol labels (random colors) over src_vol image (alpha blend)

labelfile = fullfile(dirname, 'labels.h5');
srcfile = fullfile(dirname, 'source.h5');
mkdir(dirname);

%compress seg_vol to as few labels as possible
fprintf(1, 'Compressing labels.\n');

[~, ~, new_vol] = unique(seg_vol);
new_vol = reshape(new_vol, size(seg_vol));

nsegments = max(new_vol(:));

srcmi = min(src_vol(:));
srcma = max(src_vol(:));
if srcmi == srcma
    srcma = srcmi + 1;
    fprintf(1, 'WARNING: Max and min src_vol values are the same.');
end

fprintf(1, 'Type checking.\n');

src_vol = single(src_vol);
src_vol = (src_vol - single(srcmi)) / single(srcma - srcmi);

if nsegments > 2^32
    new_vol = uint64(new_vol);
    label_type = 'uint64';
else
    new_vol = uint32(new_vol);
    label_type = 'uint32';
end

fprintf(1, 'Saving files.\n');

delete(labelfile);
h5create(labelfile, '/main', size(new_vol), 'Datatype', label_type);
h5write(labelfile, '/main', new_vol);

delete(srcfile);
h5create(srcfile, '/main', size(src_vol), 'Datatype', 'single');
h5write(srcfile, '/main', src_vol);

    