function imProb = generateMembraneProbabilities(im, forest)

%extract features
cs = 29;
ms = 3;
csHist = cs;

disp('*** FEATURE EXTRACTION ***')
tic;
fm  = membraneFeatures(im, cs, ms, csHist);
fm = reshape(fm,size(fm,1)*size(fm,2),size(fm,3));
toc;

disp('*** CLASSIFICATION ***');

disp('prediction')
tic;
[y_h,votes] = classRF_predict(double(fm), forest);
votes = votes(:,2);
votes = reshape(votes,size(im));
imProb = double(votes)/forest.ntree;
toc;

disp('done')


