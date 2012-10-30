function pairwise_match(block1, block2, direction, halo_size, outblock1, outblock2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: direction indicates the relative position of the blocks (1, 2, 3 => adjacent in X, Y, Z).
% Block1 is always closer to the 0,0,0 corner of the volume.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Change joining thresholds here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Join 1 (less joining)
auto_join_pixels = 20000; % Join anything above this many pixels overlap
minoverlap_pixels = 2000; % Consider joining all pairs over this many pixels overlap
minoverlap_dual_ratio = 0.7; % If both overlaps are above this then join
minoverlap_single_ratio = 0.9; % If either overlap is above this then join
    
% Join 2 (more joining)
% auto_join_pixels = 10000; % Join anything above this many pixels overlap
% minoverlap_pixels = 1000; % Consider joining all pairs over this many pixels overlap
% minoverlap_dual_ratio = 0.5; % If both overlaps are above this then join
% minoverlap_single_ratio = 0.8; % If either overlap is above this then join
    
disp(['Running pairwise matching']);

% Extract overlapping regions
block1_info = h5info(block1, '/labels');
block2_info = h5info(block2, '/labels');
assert(block1_info.Dataspace.Size == block2_info.Dataspace.Size);
blocksize = block1_info.Dataspace.Size;
lo_block1 = [1, 1, 1];
hi_block1 = blocksize;
lo_block2 = [1, 1, 1];
hi_block2 = blocksize;

direction = str2num(direction);
halo_size = str2num(halo_size);

% Adjust overlapping region boundaries for direction
lo_block1(direction) = blocksize(direction) - 2 * halo_size + 1;
hi_block2(direction) = 2 * halo_size;

% extract overlap
block1_overlap = h5read(block1, '/labels', lo_block1, hi_block1);
block2_overlap = h5read(block2, '/labels', lo_block2, hi_block2);

% Pack labels and convert from uint64 so we can use sparse matrices for counting overlaps
[packed1_to_orig, ia1, packed1] = unique(block1_overlap);
[packed2_to_orig, ia2, packed2] = unique(block2_overlap);

% Build the overlap count matrix
overlap_counts = sparse(packed1, packed2, ones(size(packed1)));

% Find total areas of each label within overlap region
packed1_total_areas = sum(overlap_counts, 2);
packed2_total_areas = sum(overlap_counts, 1);

% Join regions with high enough overlap counts or fractions
should_join = ((overlap_counts > auto_join_pixels) |
               ((overlap_counts > minoverlap_pixels) &
                ((overlap_counts > (minoverlap_single_ratio * packed1_total_areas)) |
                 (overlap_counts > (minoverlap_single_ratio * packed2_total_areas)) |
                 ((overlap_counts > (minoverlap_dual_ratio * packed1_total_areas)) &
                  (overlap_counts > (minoverlap_dual_ratio * packed2_total_areas))))));

% These joins will be processed at the end
to_join = [];

% Find joined regions
[r, c] = find(should_join);
for ix = 1:length(r),
    label1 = packed1_to_orig(r(ix));
    label2 = packed2_to_orig(c(ix));
    if (label1 ~= 0) & (label2 ~= 0),
       to_join = [to_join; [label1, label2]]
    end
end

% Deal with overlapping but not joined regions
if direction == 1,
  block1_face = block1_overlap(1, :, :);
  block2_face = block2_overlap(end, :, :);
elseif direction == 2,
  block1_face = block1_overlap(:, 1, :);
  block2_face = block2_overlap(:, end, :);
else,
  block1_face = block1_overlap(:, :, 1);
  block2_face = block2_overlap(:, :, end);
end;  

[r, c] = find((~ should_join) & (overlap_counts > 0));
for ix = 1:length(r),
    label1 = packed1_to_orig(r(ix));
    label2 = packed2_to_orig(c(ix));
    if (label1 ~= 0) & (label2 ~= 0),
       % XXX - This is slow
       if sum(block1_face == label1) > sum(block2_face == label2),
         % block1 wins
         new_label = label1;
       else,
         % block2 wins
         new_label = label2;
       end
       block1_overlap[(block1_overlap == label1) & (block2_overlap == label2)] = new_label;
       block2_overlap[(block1_overlap == label1) & (block2_overlap == label2)] = new_label;
    end
end

% Write output
temp1 = [outblock1, '_partial'];
temp2 = [outblock2, '_partial'];
copyfile(block1, temp1);
copyfile(block2, temp2);

% Write joins
if length(to_join) > 0,
  try,
    old_joins = h5read(temp1, '/joins');
    h5write(temp1, '/joins', [old_joins; to_join]);
  catch,
    h5write(temp1, '/joins', to_join);
  end
end

% write out new overlap regions
h5write(temp1, '/labels', block1_overlap, lo_block1, hi_block1);
h5write(temp2, '/labels', block2_overlap, lo_block2, hi_block2);

% move temporaries into place
movefile(temp1, outblock1);
movefile(temp2, outblock2);
