%Test GPU bagging

k_bagsorter = parallel.gpu.CUDAKernel('train3.ptx', 'train3.cu', '_Z16randomsamplesortPKfiiiPKiS2_iPiiyyS3_');

seed = 0;
sequencestart = 0;

x = [0.01:0.01:1; (1:1:100) .*  rand(1,100); (201:1:300) .*  rand(1,100); (100:-1:1) .*  rand(1,100); rand(1,100); rand(1,100).*1000]';
classes = [ones(1,30),ones(1,70)*2];
n = size(x,1);
mdim = size(x,2);
nclass = int32(2);
nsamples = int32([10 16]);
samplefrom = int32(zeros(1,nclass));
totsamples = sum(nsamples);
nbags = 8;
for c = 1:nclass
    samplefrom(c) = sum(classes==c);
end

classindex = -ones(max(samplefrom)*nclass, 1, 'int32');

cioffset = 0;
for c = 1:nclass
    classindex((1:samplefrom(c))+cioffset) = find(classes==c)'-1;
    cioffset = cioffset + samplefrom(c);
end

dev_bagspace = gpuArray(-ones([totsamples, nbags], 'int32'));
dev_tempbag = gpuArray(-ones([totsamples, nbags], 'int32'));
dev_baggedx = gpuArray(zeros(sum(nsamples)*mdim, nbags, 'single'));

threadsPerBlock = 32;
gridSizeX = ceil(nbags/threadsPerBlock);

%Get kernel for prediction
k_bagsorter.ThreadBlockSize = threadsPerBlock;
k_bagsorter.GridSize = gridSizeX;

[dev_bagspace dev_tempbag dev_baggedx] = feval(k_bagsorter, x, n, mdim, nclass, nsamples, ...
    samplefrom, totsamples, dev_bagspace, nbags, seed, sequencestart, ...
    dev_tempbag, classindex, dev_baggedx);

bags = gather(dev_bagspace);
baggedx = gather(dev_baggedx);
tempbag = gather(dev_tempbag);

clear dev_bagspace;
clear dev_tempbag;
clear dev_baggedx;
clear k_bagger;
reset(gpuDevice(1));

%Check sort results like this
checkn = 1;
featurei = 3;
baggedx(bags(1:totsamples,checkn)+1+totsamples*featurei,checkn)

bags