%draws a frame around an image with given value
% imf = drawFrame(im, width, val)
function imf = drawFrame(im, width, val)
  imf = im;
  imf(1:width,:) = val;
  imf(end-width+1:end,:) = val;
  imf(:,1:width) = val;
  imf(:,end-width+1:end) = val;

  