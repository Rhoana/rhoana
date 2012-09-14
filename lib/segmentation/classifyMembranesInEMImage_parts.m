function classifyMembranesInEMImage_parts(imageNumber)
imageNumber = str2double(imageNumber)

%extract features
cs = 29;
ms = 3;
csHist = cs;

imgNames = dir('*_image*.png');

name = imgNames(imageNumber).name;
disp(name);

if exist(strcat(name(1:end-4),'_imProb.mat'))
    return
end

disp('*** FEATURE EXTRACTION ***')
tic;
im = imread(imgNames(imageNumber).name);

fm  = membraneFeatures(im, cs, ms, csHist);
fm = reshape(fm,size(fm,1)*size(fm,2),size(fm,3));
toc;

disp('*** CLASSIFICATION ***');
load forest.mat

disp('prediction')
tic;
[y_h,votes] = classRF_predict(double(fm), forest);

votes = votes(:,2);
votes = reshape(votes,size(im));
imProb = double(votes)/max(votes(:));
toc;
disp('writing to disk')
tic;
save(strcat(name(1:end-4),'_imProb.mat'),'imProb');
toc;
disp('done')


