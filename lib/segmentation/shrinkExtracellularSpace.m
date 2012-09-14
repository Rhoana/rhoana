%test = shrinkExtracellularSpace(im)
function test = shrinkExtracellularSpace(im)
im = uint32(im);

%this function is in the git repository under
%previousWork/Bjoern/cpp/src/andres/mex
%compile with: 
%mex -ID:\connectome\PreviousWork\Bjoern\cpp\include\andres andres_seeded_region_growing.cxx

andres_seeded_region_growing(zeros(1,size(im,1),size(im,2),'uint8'),...
    reshape(im,[1 size(im,1) size(im,2)]));

test = double(im);

% test = im;
% se = strel('disk',1);
% sumBefore = 0;
% while ismember(0,test)
%     [~,~,mag] = gradientImg(test,1);
%     tmp = imdilate(test,se);
%     ind = mag>0 & test==0;
%     test(ind) = tmp(ind);
%     if sumBefore == sum(sum(test==0))
%         break;
%     end
%     sumBefore = sum(sum(test==0));
% end