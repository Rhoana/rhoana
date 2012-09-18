#include "mex.h"

#include "vision/seeded-region-growing.hxx"

template<class T>
void helper(
    const mxArray* elevation,
    const mxArray* seeds
)
{
    andres::View<unsigned char> elevationView(mxGetDimensions(elevation), mxGetDimensions(elevation) + 3, 
        static_cast<unsigned char*>(mxGetData(elevation)));
    andres::View<T> seedsView(mxGetDimensions(seeds), mxGetDimensions(seeds) + 3, 
        static_cast<T*>(mxGetData(seeds)));
    andres::vision::seededRegionGrowing(elevationView, seedsView);
}

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]
)
{
    if(nrhs != 2) {
        mexErrMsgTxt("incorrect number of input parameters. expecting 2.\n");
    }
    if(nlhs != 0) {
        mexErrMsgTxt("incorrect number of return parameters. expecting none.\n");
    }
    if(mxGetNumberOfDimensions(prhs[0]) != 3) { 
        mexErrMsgTxt("elevation (parameter 1) is not a 3-dimensional.\n");
    }
    if(mxGetNumberOfDimensions(prhs[1]) != 3) {
        mexErrMsgTxt("seeds (parameter 2) is not a 3-dimensional.\n");
    }
    for(size_t j=0; j<3; ++j) {
        if(mxGetDimensions(prhs[0])[j] != mxGetDimensions(prhs[1])[j]) {
            mexErrMsgTxt("shape mismatch between elevation (parameter 1) and seeds (parameter 2).\n");
        }
    }
    if(mxGetClassID(prhs[0]) != mxUINT8_CLASS) {
        mexErrMsgTxt("data type not supported for elevation (parameter 1). expecting uint8.\n");
    }

    switch(mxGetClassID(prhs[1])) {
    case mxUINT8_CLASS:
        helper<unsigned char>(prhs[0], prhs[1]);
        break;
    case mxINT8_CLASS:
        helper<signed char>(prhs[0], prhs[1]);
        break;
    case mxUINT16_CLASS:
        helper<unsigned short>(prhs[0], prhs[1]);
        break;
    case mxINT16_CLASS:
        helper<short>(prhs[0], prhs[1]);
        break;
    case mxUINT32_CLASS:
        helper<unsigned int>(prhs[0], prhs[1]);
        break;
    case mxINT32_CLASS:
        helper<int>(prhs[0], prhs[1]);
        break;
    case mxUINT64_CLASS:
        helper<unsigned long>(prhs[0], prhs[1]);
        break;
    case mxINT64_CLASS:
        helper<long>(prhs[0], prhs[1]);
        break;
    default:
        mexErrMsgTxt("seed map (parameter 2) data type not supported. expecting an integer type.\n");
        break;
    }
}
