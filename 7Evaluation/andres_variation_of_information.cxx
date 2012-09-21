#include "mex.h"

#include "partition-comparison.hxx"

template<class T>
double
matlabVIHelper(
    const mxArray* array0,
    const mxArray* array1
)
{
    if(mxGetClassID(array0) != mxGetClassID(array1)) {
        mexErrMsgTxt("The two labelings do not have the same data type.\n");
    }

    if(mxGetNumberOfElements(array0) != mxGetNumberOfElements(array1)) {
        mexErrMsgTxt("The two labelings do not have the same size.\n");
    }

    const size_t n = mxGetNumberOfElements(array0);
    const T* p0 = static_cast<const T*>(mxGetData(array0));
    const T* p1 = static_cast<const T*>(mxGetData(array1));
    return andres::variationOfInformation(p0, p0 + n, p1, true); // ignores default label 0
}

void mexFunction(
    int nlhs,
    mxArray *plhs[],
    int nrhs,
    const mxArray *prhs[]
)
{
    if(nrhs != 2) {
        mexErrMsgTxt("Incorrect number of parameters. Expecting two.\n");
    }
    if(nlhs > 1) {
        mexErrMsgTxt("Incorrect number of return parameters. Expecting one.\n");
    }

    double r;
    switch(mxGetClassID(prhs[0])) {

    case mxUINT8_CLASS:
        r = matlabVIHelper<unsigned char>(prhs[0], prhs[1]);
        break;
    case mxINT8_CLASS:
        r = matlabVIHelper<char>(prhs[0], prhs[1]);
        // not signed char according to MATLAB reference doc
        break;

    case mxUINT16_CLASS:
        r = matlabVIHelper<unsigned short>(prhs[0], prhs[1]);
        break;
    case mxINT16_CLASS:
        r = matlabVIHelper<short>(prhs[0], prhs[1]);
        // not signed char according to MATLAB reference doc
        break;

    case mxUINT32_CLASS:
        r = matlabVIHelper<unsigned int>(prhs[0], prhs[1]);
        break;
    case mxINT32_CLASS:
        r = matlabVIHelper<int>(prhs[0], prhs[1]);
        // not signed char according to MATLAB reference doc
        break;

    case mxUINT64_CLASS:
        r = matlabVIHelper<unsigned long long>(prhs[0], prhs[1]);
        break;
    case mxINT64_CLASS:
        r = matlabVIHelper<long long>(prhs[0], prhs[1]);
        // not signed char according to MATLAB reference doc
        break;

    case mxSINGLE_CLASS:
        r = matlabVIHelper<float>(prhs[0], prhs[1]);
        break;
    case mxDOUBLE_CLASS:
        r = matlabVIHelper<double>(prhs[0], prhs[1]);
        break;

    default:
        mexErrMsgTxt("Data type currently not supported. Stick with uint32.\n");
        break;
    }

    plhs[0] = mxCreateDoubleScalar(r);
}
