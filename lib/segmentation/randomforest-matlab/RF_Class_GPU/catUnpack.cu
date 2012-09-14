#define NODE_TERMINAL -1
#define NODE_TOSPLIT  -2
#define NODE_INTERIOR -3

__global__ void catUnpack(int *nodestatus, float *xbestsplit, int *bestvar,
		      int *cat, int maxcat, int *cbestsplit, int maxTreeSize)
{
    int threadi = threadIdx.x;
	int treeOffset = threadi*maxTreeSize;

    int i, j;
	unsigned int npack;

    /* decode the categorical splits */
    if (maxcat > 1) {
        for (i = treeOffset; i < (maxTreeSize + treeOffset); ++i) {
			//Init this tree node to zero for all categories
			for (j = 0; j < maxcat; ++j) {
				cbestsplit[i + j]=0;
			}
			//Ignore terminal nodes
            if (nodestatus[i] != NODE_TERMINAL) {
				//Do we have a bestvar with more than one category?
                if (cat[bestvar[i] - 1] > 1) {
                    npack = (unsigned int) xbestsplit[i];
                    /* unpack `npack' into bits */
                    for (j = 0; npack; npack >>= 1, ++j) {
                        cbestsplit[j + i*maxcat] = npack & 01;
                    }
                }
            }
        }
    }
}