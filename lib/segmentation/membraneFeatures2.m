% featureMatrix  = membraneFeatures(image, contextSize,
% membraneThickness, contextSizeHistogram, hessianSigma)
function [fm]  = membraneFeatures2(im, cs, ms, csHist)

n_features = 108;
upto_feature = 1;

%normalize image contrast
im = norm01(im);
imOrig = im;

d = zeros(cs,cs);
s = round(cs / 2);
d(:,s-ms:s+ms) = 1;
d = single(d);

fm = zeros(size(im,1), size(im,2), n_features, 'single');
fm(:,:,1) = im;

im = adapthisteq(im);

rot = filterImageWithMembraneTemplateRotated(im, d);
im = single(im);


fm(:,:,2) = rot(:,:,1);
fm(:,:,3) = rot(:,:,2);
fm(:,:,4) = rot(:,:,3);
fm(:,:,5) = rot(:,:,4);
fm(:,:,11) = rot(:,:,5);
fm(:,:,12) = rot(:,:,6);
fm(:,:,13) = rot(:,:,7);
fm(:,:,14) = rot(:,:,8);

rot = shiftdim(rot,2);

rotMin = min(rot);
fm(:,:,6) = shiftdim(rotMin,1);
clear rotMin;

rotMax = max(rot);
fm(:,:,7) = shiftdim(rotMax,1);
clear rotMax;

rotMean = mean(rot);
fm(:,:,8) = shiftdim(rotMean,1);
clear rotMean;

rotVar = var(rot);
fm(:,:,9) = shiftdim(rotVar,1);
clear rotVar;

rotMedian = median(rot);
fm(:,:,10) = shiftdim(rotMedian,1);
clear rotMedian;

clear rot;

upto_feature = upto_feature + 14;

disp('histogram');
csHalf = floor(csHist/2);
for i=1:length(im(:))
    [r,c] = ind2sub(size(im),i);
    if r < cs | c < cs | r > size(im,1)-cs | c > size(im,2)-cs
        continue
    else
        sub = im(r-csHalf:r+csHalf, c-csHalf:c+csHalf);
        sub = norm01(sub) * 100;
        [m,v,h] = meanvar(sub);
        fm(r,c,17:26) = h;
        fm(r,c,27) = m;
        fm(r,c,28) = v;
    end
end

upto_feature = upto_feature + 14;

%rotMax - rotMin
fm(:,:,upto_feature) = fm(:,:,7) - fm(:,:,6);
upto_feature = upto_feature + 1;

[eig1, eig2, cw] = structureTensorImage2(im, 1, 1);
%fixing nans in the division, eig1 is zero there anyways.
eig2(find(eig2==0)) = 1;

[gx,gy,mag] = gradientImg(im,1);
clear gx
clear gy

disp('for loop');

%used to be 20
%Nancy: 1:1:10
%was: 1:2:20
for i=1:1:10
    ims = imsmooth(double(im),i);
    fm(:,:,upto_feature) = ims;
    upto_feature = upto_feature + 1;
    fm(:,:,upto_feature) = imsmooth(eig1./eig2,i);
    upto_feature = upto_feature + 1;
    fm(:,:,upto_feature) = imsmooth(mag,i);
    upto_feature = upto_feature + 1;
    [gx,gy,mags] = gradientImg(im,i);
    fm(:,:,upto_feature) = mags;
    upto_feature = upto_feature + 1;
    
    for j=1:2:i-2
        fm(:,:,upto_feature) = ims - single(imsmooth(double(im),j));
        upto_feature = upto_feature + 1;
    end
end

smoothFM = fm(:,:,upto_feature-10:upto_feature-1);
%90 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
fm(:,:,upto_feature) = shiftdim(var(shiftdim(smoothFM,2)),1);
upto_feature = upto_feature + 1;
%91
fm(:,:,upto_feature) = norm01(imsmooth(double(im),2) - ...
    imsmooth(double(im),50));
upto_feature = upto_feature + 1;
%92
fm(:,:,upto_feature) = im;
upto_feature = upto_feature + 1;
   
for s=[0 1 2 3 4 6 8 10]
    [eig1, eig2, ~, ~, ~, ~, ~, ~] = eigImg(imsmooth(imOrig,s));
    fm(:,:,upto_feature) = eig1;
    upto_feature = upto_feature + 1;
    fm(:,:,upto_feature) = eig2;
    upto_feature = upto_feature + 1;
end

fprintf(1, 'Total features = %d\n', upto_feature-1);

disp('Feature extraction finished')

