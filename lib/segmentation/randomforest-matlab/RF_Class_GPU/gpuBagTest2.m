%Test GPU bagging

k_bagger = parallel.gpu.CUDAKernel('train2.ptx', 'train2.cu', '_Z12randomsampleiPKiS0_iPiiyy');

seed = 0;
sequencestart = 0;

nclass = 3;
nsamples = [10 6 9];
samplefrom = [100 200 6];
maxnsamples = max(nsamples);
nbags = 5;

dev_bagspace = gpuArray(-ones([maxnsamples*nclass, nbags], 'int32'));

threadsPerBlock = 32;
gridSizeX = ceil(nbags/threadsPerBlock);

%Get kernel for prediction
k_bagger.ThreadBlockSize = threadsPerBlock;
k_bagger.GridSize = gridSizeX;

dev_bagspace = feval(k_bagger, nclass, nsamples, samplefrom, maxnsamples, dev_bagspace, nbags, seed, sequencestart);

bags = gather(dev_bagspace);

clear dev_bagspace;
clear k_bagger;
reset(gpuDevice(1));

bags