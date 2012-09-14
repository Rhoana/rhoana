% wrapper function for random forest prediction on GPU

function [votes] = gpu_classRF_predict(X,model)
    
	votes = gpu_predict(X',model.nrnodes,model.ntree,model.xbestsplit,model.treemap,model.nodestatus,model.nodeclass,model.bestvar,model.nclass);
	%keyboard
    votes = votes';
    
end
    