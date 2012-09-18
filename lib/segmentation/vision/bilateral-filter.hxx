#pragma once
#ifndef ANDRES_VISION_BILATERAL_HXX
#define ANDRES_VISION_BILATERAL_HXX

#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#define mymax(a,b) (((a) > (b)) ? (a) : (b))
#define mymin(a,b) (((a) < (b)) ? (a) : (b))

namespace andres {
namespace vision {

// C-like implementation of a 3d bilateral filter
//
// pre-conditions:
// - enough memory has been allocated for out, e.g. using
//   malloc(nx*ny*nz*sizeof(VoxelType))
// - out has been filled with zeros
//
template<class VoxelType>
void bilateral3(
    const VoxelType* in,
    const ptrdiff_t nx,
    const ptrdiff_t ny,
    const ptrdiff_t nz,
    const VoxelType spatialScale,
    const VoxelType intensityScale,
    const ptrdiff_t radius,
    VoxelType* out
) {
    const ptrdiff_t size  = 2 * radius + 1;
    const ptrdiff_t size2 = size * size;
    const ptrdiff_t size3 = size2 * size;
    const size_t    M     = size3 * sizeof(VoxelType);
    const ptrdiff_t nxy   = nx * ny;
    const VoxelType rho   = intensityScale * intensityScale;
    const VoxelType p     = -2 * spatialScale * spatialScale;

    VoxelType *gaussFilter = (VoxelType*) malloc(M);
    VoxelType *bilateralFilter = (VoxelType*) malloc(M);

    // gauss filter pre-computation
    for(ptrdiff_t z=-radius; z<=radius; ++z)
    for(ptrdiff_t y=-radius; y<=radius; ++y)
    for(ptrdiff_t x=-radius; x<=radius; ++x) {
        gaussFilter[(x+radius) + size*(y+radius) + size2*(z+radius)]
            = exp( (x*x + y*y + z*z)/p ) ;
    }

    // diffusion (push)
    for(ptrdiff_t z0=0; z0<nz; ++z0)
    for(ptrdiff_t y0=0; y0<ny; ++y0)
    for(ptrdiff_t x0=0; x0<nx; ++x0) {
        ptrdiff_t j0 = x0 + nx*y0 + nxy*z0;
        VoxelType sum = 0;
        // ***
        // compute filter mask
        for(ptrdiff_t z=mymax(z0-radius, 0); z<=mymin(z0+radius, nz-1); ++z)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y)
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x) {
            ptrdiff_t j = x + nx*y + nxy*z;
            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t zr = z - z0 + radius;
            ptrdiff_t q = xr + size*yr + size2*zr;
            VoxelType val = gaussFilter[q] * ( 1.0f / (1.0f + (in[j]-in[j0])*(in[j]-in[j0])/rho) );

            bilateralFilter[q] = val;
            sum += val;
        }
        // normalize filter mask
        for(ptrdiff_t z=mymax(z0-radius, 0); z<=mymin(z0+radius, nz-1); ++z)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y)
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x) {
            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t zr = z - z0 + radius;
            ptrdiff_t q = xr + size*yr + size2*zr;
            bilateralFilter[q] /= sum;
        }

        // ***
        // diffuse
        for(ptrdiff_t z=mymax(z0-radius, 0); z<=mymin(z0+radius, nz-1); ++z)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y)
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x) {
            ptrdiff_t j = x + nx*y + nxy*z;

            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t zr = z - z0 + radius;
            ptrdiff_t q = xr + size*yr + size2*zr;

            out[j0] += (in[j] * bilateralFilter[q]);
        }
    }

    // clean-up
    free(gaussFilter);
    free(bilateralFilter);
}

} // namespace vision
} // namespace andres

#endif // #ifndef ANDRES_VISION_BILATERAL_HXX
