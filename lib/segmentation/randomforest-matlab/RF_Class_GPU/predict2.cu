#define NODE_TERMINAL -1
#define NODE_TOSPLIT  -2
#define NODE_INTERIOR -3

__global__ void predictKernel(const float *x, int n, int mdim, const int *treemap,
		      const int *nodestatus, const float *xbestsplit,
		      const int *bestvar, const int *nodeclass,
		      int nclass,
			  int ntree, int *countts, int maxTreeSize)
		      //int *jts,
			  //int *nodex,
{
	int idx = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);

	//Make sure we don't overrun
	if (idx < n) {
		int m, k, treei, treeOffset1, treeOffset2;

		//Repeat for each tree - this way only one thread writes to any point in the vote output array

		for (treei = 0; treei < ntree; ++treei) {
		//for (treei = 0; treei < ntree; ++treei) {
			treeOffset1 = treei*maxTreeSize;
			treeOffset2 = treei*2*maxTreeSize;
			k = 0;

			while (nodestatus[treeOffset1 + k] != NODE_TERMINAL) {
				m = bestvar[treeOffset1 + k] - 1;
				//Split by a numerical predictor
				k = (x[idx + n * m] <= xbestsplit[treeOffset1 + k]) ?
					treemap[treeOffset2 + k * 2] - 1 : treemap[treeOffset2 + 1 + k * 2] - 1;
			}
			//We found the terminal node: assign class label
			//jts[chunki + treei] = nodeclass[treeOffset + k];
			//nodex[chunki + treei] = k + 1;
			countts[idx * nclass + nodeclass[treeOffset1 + k] - 1] += 1;
		}
	}

}