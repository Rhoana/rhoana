__global__ void add( double * v1, const double * v2, int chunksize ) 
{
    int chunkstart = threadIdx.x*chunksize;
	int chunkend = (threadIdx.x+1)*chunksize;
	int i = 0;
	for (i = chunkstart; i < chunkend; ++i) {
		v1[i] += v2[i];
	}
}