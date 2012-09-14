function [votes] = gpu_predict(X,nrnodes,ntree,xbestsplit,treemap,nodestatus,nodeclass,bestvar,nclass)

threadsPerBlock = 32;
gridSizeX = 1024;
gridSizeY = ceil(size(X,2)/threadsPerBlock/gridSizeX);

%Copy data to gpu
dev_x = gpuArray(single(X));
dev_treemap = gpuArray(treemap);
dev_nodestatus = gpuArray(nodestatus);
dev_xbestsplit = gpuArray(single(xbestsplit));
dev_bestvar = gpuArray(bestvar);
dev_nodeclass = gpuArray(nodeclass);
%dev_jts = gpuArray(zeros(nrnodes*ntree, 1, 'int32'));
%dev_nodex = gpuArray(zeros(nrnodes*ntree, 1, 'int32'));
dev_countts = gpuArray(zeros(nclass, size(X,2), 'int32'));

%Get kernel for prediction
k_predict = parallel.gpu.CUDAKernel('predict.ptx', 'predict.cu');
k_predict.ThreadBlockSize = threadsPerBlock;
k_predict.GridSize = [gridSizeX, gridSizeY];

dev_votes = feval(k_predict, dev_x, size(X,2), size(X,1), dev_treemap, dev_nodestatus, ...
    dev_xbestsplit, dev_bestvar, dev_nodeclass, nclass, ...
    ...%dev_jts, dev_nodex,
    ntree, dev_countts, nrnodes);

%Y_hat = gather(dev_nodex);
%prediction_per_tree = gather(dev_jts);
votes = gather(dev_votes);

end
