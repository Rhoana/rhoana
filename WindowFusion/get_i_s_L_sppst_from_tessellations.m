function i_s_L_sppst  = get_i_s_L_sppst_from_tessellations(i_thin_hard_bdries, area_to_clean)

% 50th of a screen's axis
nR                   = size(i_thin_hard_bdries, 1);
nC                   = size(i_thin_hard_bdries, 2);
n_frames_to_process  = size(i_thin_hard_bdries, 3);


i_s_L_sppst = zeros([nR nC n_frames_to_process], 'uint32');

for frame_ix = 1:n_frames_to_process

   i_bdry_stack             = any(i_thin_hard_bdries(:, :, frame_ix, :), 4); 
   i_bdry_stack(1:end,1)    = 1;    i_bdry_stack(1,1:end) = 1;
   i_bdry_stack(1:end,end)  = 1;    i_bdry_stack(end,1:end) = 1;           
         
   %% Clean the image (remove tiny segments and isolated edges)
   i_bdry = i_bdry_stack; 

   i_result = bwareaopen(~i_bdry, area_to_clean, 4);
   
%    figure(11);
%    imshow(i_result, []);
   CC = bwconncomp(i_result,8);

   STATS = regionprops(CC, 'Euler');

   which_edge_elements_to_remove = [STATS(:).EulerNumber] < 0;
   CC.PixelIdxList(which_edge_elements_to_remove) = [];
   CC.NumObjects = CC.NumObjects - sum(which_edge_elements_to_remove);
   CC.Connectivity = 4;

   i_euler_cleaned = labelmatrix(CC);
   i_euler_cleaned = i_euler_cleaned ~= 0;
   i_bdry          = i_euler_cleaned;
   i_bdry          = ~i_bdry;
   
   %%
%    figure(12);
%    imshow(i_bdry_stack, [])
%    figure(13);
%    imshow(i_bdry, []);
   %%
   i_L                          = bwconncomp(~i_bdry, 4);
   %%
   i_s_L_sppst(:,:,frame_ix)    = labelmatrix(i_L);   
end


end