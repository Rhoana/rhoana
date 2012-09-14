%Test GPU bagging
%clear k_bagger;
%reset(gpuDevice(1));

k_train = parallel.gpu.CUDAKernel('train.ptx', 'train.cu');

seed = 0;
sequencestart = 0;

% nclass = 3;
% nsamples = [10 6 9];
% samplefrom = [100 200 6];
% maxnsamples = max(nsamples);
% nbags = 500;

nclass = int32(2);
nsamples = int32([10 16]);
samplefrom = int32(zeros(1,nclass));
ntree = int32(5);
%maxTreeSize = int32(20001);
maxTreeSize = int32(10);
mtry = int32(2);
nodeStopSize = int32(1);

x = single([0.01:0.01:1; (1:1:100) .*  rand(1,100); (201:1:300) .*  rand(1,100); (100:-1:1) .*  rand(1,100); rand(1,100); rand(1,100).*1000]');
classes = int32([ones(1,30),ones(1,70)*2]);

for c = 1:nclass
    samplefrom(c) = sum(classes==c);
end

maxnsamples = max(nsamples);
classindex = -ones(max(samplefrom)*nclass, 1, 'int32');

cioffset = 0;
for c = 1:nclass
    classindex((1:samplefrom(c))+cioffset) = find(classes==c)'-1;
    cioffset = cioffset + samplefrom(c);
end

tic
dev_bagspace = gpuArray(-ones([maxnsamples*nclass, ntree], 'int32'));
dev_tempbag = gpuArray(-ones([maxnsamples*nclass, ntree], 'int32'));

dev_treemap = gpuArray(zeros(maxTreeSize, ntree*2, 'int32'));
dev_nodestatus = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_xbestsplit = gpuArray(zeros(maxTreeSize, ntree, 'single'));
%dev_nbestsplit = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
%dev_bestgini = gpuArray(zeros(maxTreeSize, ntree, 'single'));
dev_bestvar = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_nodeclass = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_ndbigtree = gpuArray(zeros(ntree, 2, 'int32'));
dev_nodestart = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_nodepop = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_classpop = gpuArray(zeros(maxTreeSize*nclass, ntree, 'int32'));
dev_classweights = gpuArray(ones(nclass, ntree, 'single'));
dev_weight_left = gpuArray(zeros(nclass, ntree, 'int32'));
dev_weight_right = gpuArray(zeros(nclass, ntree, 'int32'));
dev_dimtemp = gpuArray(zeros(size(x,2), ntree, 'int32'));
toc

threadsPerBlock = 32;
gridSizeX = ceil(double(ntree)/threadsPerBlock);

%Get kernel for prediction
k_train.ThreadBlockSize = threadsPerBlock;
k_train.GridSize = gridSizeX;

tic
[dev_treemap, dev_nodestatus, dev_xbestsplit, ...%dev_nbestsplit, dev_bestgini ...
    dev_bestvar dev_nodeclass dev_nbigtree ...
    dev_nodestart, dev_nodepop, ...
    dev_classpop, dev_classweights, ...
    dev_weight_left, dev_weight_right, ...
    dev_dimtemp, dev_bagspace, dev_tempbag ...
    ] = feval(k_train, x, size(x,1), size(x,2), nclass, ...
    classes, classindex, ...
    nsamples, samplefrom, ...
    maxnsamples, seed, sequencestart, ...
    ntree, maxTreeSize, mtry, nodeStopSize, ...
	dev_treemap, dev_nodestatus, dev_xbestsplit, dev_nbestsplit, dev_bestgini, ...
	dev_bestvar, dev_nodeclass, dev_ndbigtree, ...
    dev_nodestart, dev_nodepop, ...
    dev_classpop, dev_classweights, ...
    dev_weight_left, dev_weight_right, ...
    dev_dimtemp, dev_bagspace, dev_tempbag);
toc

tic
bags = gather(dev_bagspace);
treemap = gather(dev_treemap);
nodestatus = gather(dev_nodestatus);
xbestsplit = gather(dev_xbestsplit);
bestvar = gather(dev_bestvar);
nodeclass = gather(dev_nodeclass);
nbigtree = gather(dev_nbigtree);

nodestart = gather(dev_nodestart);
nodepop = gather(dev_nodepop);
classpop = gather(dev_classpop);
classweights = gather(dev_classweights);
weight_left = gather(dev_weight_left);
weight_right = gather(dev_weight_right);
dimtemp = gather(dev_dimtemp);
tempbag = gather(dev_tempbag);

%Optional buffers - just for debugging
%nbestsplit = gather(dev_nbestsplit);
%bestgini = gather(dev_bestgini);
toc


%clear dev_bagspace;
%clear k_bagger;
%reset(gpuDevice(1));

%bags
nodeclass