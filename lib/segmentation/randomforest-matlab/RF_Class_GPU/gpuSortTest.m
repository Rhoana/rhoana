%Test GPU bagging

k_bagsorter = parallel.gpu.CUDAKernel('train.ptx', 'train.cu', '_Z16randomsamplesortPKfiiiPKiS2_iPiiyyS3_');

seed = 0;
sequencestart = 0;

x = [1:1:500;500:-1:1;randn(1,500)]';
classes = int32([ones(1,100),ones(1,200)*2,ones(1,6)*3]);
n = size(x,1);
mdim = size(x,2);

% nclass = 3;
% nsamples = [10 6 9];
% samplefrom = [100 200 6];
% maxnsamples = max(nsamples);
% nbags = 9;

% nclass = 2;
% nsamples = [16 16];
% samplefrom = [100 200];
% maxnsamples = max(nsamples);
% nbags = 500;

x = single([0.01:0.01:1; 1:1:100; 201:1:300]');
classes = int32([ones(1,30),ones(1,70)*2]);
n = size(x,1);
mdim = size(x,2);
nclass = int32(2);
nsamples = int32([10 16]);
samplefrom = int32(zeros(1,nclass));
maxnsamples = max(nsamples);
nbags = 5;
for c = 1:nclass
    samplefrom(c) = sum(classes==c);
end

classindex = -ones(max(samplefrom)*nclass, 1, 'int32');

cioffset = 0;
for c = 1:nclass
    classindex((1:samplefrom(c))+cioffset) = find(classes==c)'-1;
    cioffset = cioffset + samplefrom(c);
end

dev_bagspace = gpuArray(-ones([maxnsamples*nclass, nbags], 'int32'));
dev_tempbag = gpuArray(-ones([maxnsamples*nclass, nbags], 'int32'));

threadsPerBlock = 32;
gridSizeX = ceil(nbags/threadsPerBlock);

%Get kernel for prediction
k_bagsorter.ThreadBlockSize = threadsPerBlock;
k_bagsorter.GridSize = gridSizeX;

dev_bagspace = feval(k_bagsorter, x, n, mdim, nclass, nsamples, ...
    samplefrom, maxnsamples, dev_bagspace, nbags, seed, sequencestart, ...
    dev_tempbag, classindex);

bags = gather(dev_bagspace);

clear dev_bagspace;
clear k_bagger;
reset(gpuDevice(1));

bags