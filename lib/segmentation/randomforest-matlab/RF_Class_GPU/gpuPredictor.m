%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GPU Implementation of Random Forest Classifier - Prediction
%v0.1
%Seymour Knowles-Barley
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%use like this:
%   [load a random forest data structure "forest"]
%   predictor = gpuPredictor(forest);
%   [for loop]
%       [load features "fm"]
%       votes = predictor.predict(fm);
%       [do something with votes]
%   [end for]
%   clear predictor
%IMPORTANT: remember to clear GPU resources are released.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


classdef gpuPredictor < handle
    properties (Hidden)
        dev_treemap
        dev_nodestatus
        dev_xbestsplit
        dev_bestvar
        dev_nodeclass
        nrnodes
        ntree
        nclass
        k_predict
    end
    methods
        function obj = gpuPredictor(model)
            
           %Copy data to gpu
            obj.dev_treemap = gpuArray(model.treemap);
            obj.dev_nodestatus = gpuArray(model.nodestatus);
            obj.dev_xbestsplit = gpuArray(single(model.xbestsplit));
            obj.dev_bestvar = gpuArray(model.bestvar);
            obj.dev_nodeclass = gpuArray(model.nodeclass);
            obj.nrnodes = model.nrnodes;
            obj.ntree = model.ntree;
            obj.nclass = model.nclass;

            %Get the kernel
            obj.k_predict = parallel.gpu.CUDAKernel('predict.ptx', 'predict.cu');
            
        end
        
        function [votes] = predict(obj, X)
            
            threadsPerBlock = 32;
            gridSizeX = 1024;
            gridSizeY = ceil(size(X,1)/threadsPerBlock/gridSizeX);
            
            dev_x = gpuArray(single(X)');
            dev_countts = gpuArray(zeros(obj.nclass, size(X,1), 'int32'));
            
            %Get kernel for prediction
            obj.k_predict.ThreadBlockSize = threadsPerBlock;
            obj.k_predict.GridSize = [gridSizeX, gridSizeY];
            
            dev_votes = feval(obj.k_predict, dev_x, size(X,1), size(X,2), obj.dev_treemap, obj.dev_nodestatus, ...
                obj.dev_xbestsplit, obj.dev_bestvar, obj.dev_nodeclass, obj.nclass, ...
                ...%obj.dev_jts, obj.dev_nodex,
                obj.ntree, dev_countts, obj.nrnodes);
            
            %TODO: Set up for asynchronys calls
            %requires Matlab 2012
            %wait(gpudev);
            
            %Y_hat = gather(dev_nodex);
            %prediction_per_tree = gather(dev_jts);
            votes = gather(dev_votes)';
            
        end
        
        function delete(obj)
            try
                clear obj.dev_treemap;
                clear obj.dev_nodestatus;
                clear obj.dev_xbestsplit;
                clear obj.dev_bestvar;
                clear obj.dev_nodeclass;
                clear obj.k_predict;
                reset(gpuDevice(1));
            catch err
                fprintf(1, 'Error unbinding from GPU: %s\n', err.message);
            end
        end
        
    end
end