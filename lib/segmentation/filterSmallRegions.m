% imFiltered = filterSmallRegions(im, regionSize)
function imFiltered = filterSmallRegions(im, regionSize)

%bw = bwconncomp(im,4);
s = regionprops(bwlabel(im,4),'Area','PixelIdxList');
  for i=1:length(s)
    if s(i).Area < regionSize
      im(s(i).PixelIdxList) = 0;
    end
  end
imFiltered = im;
