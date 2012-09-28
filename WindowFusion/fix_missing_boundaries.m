function [ rawImage ] = fix_missing_boundaries(rawImage)
  shiftedImage = circshift(rawImage,1);      %# Shift image down one row
  index = (rawImage ~= shiftedImage) & ...   %# A logical matrix with ones where
          rawImage & shiftedImage;           %#   up-down neighbors differ and
                                             %#   neither is black
  rawImage(index) = 0;                       %# Set those pixels to black
  shiftedImage = circshift(rawImage,[0 1]);  %# Shift image right one column
  index = (rawImage ~= shiftedImage) & ...   %# A logical matrix with ones where
          rawImage & shiftedImage;           %#   up-down neighbors differ and
                                             %#   neither is black
  rawImage(index) = 0;                       %# Set those pixels to black

end

