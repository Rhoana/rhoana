/* file:        siftdescriptor
** author:      Andrea Vedaldi
** description: Compute SIFT descriptors
**/

/* AUTORIGHTS
Copyright (c) 2006 The Regents of the University of California.
All Rights Reserved.

Created by Andrea Vedaldi
UCLA Vision Lab - Department of Computer Science

Permission to use, copy, modify, and distribute this software and its
documentation for educational, research and non-profit purposes,
without fee, and without a written agreement is hereby granted,
provided that the above copyright notice, this paragraph and the
following three paragraphs appear in all copies.

This software program and documentation are copyrighted by The Regents
of the University of California. The software program and
documentation are supplied "as is", without any accompanying services
from The Regents. The Regents does not warrant that the operation of
the program will be uninterrupted or error-free. The end-user
understands that the program was developed for research purposes and
is advised not to rely exclusively on the program for any reason.

This software embodies a method for which the following patent has
been issued: "Method and apparatus for identifying scale invariant
features in an image and use of same for locating an object in an
image," David G. Lowe, US Patent 6,711,293 (March 23,
2004). Provisional application filed March 8, 1999. Asignee: The
University of British Columbia.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

/*
  REMARKS. The use of strcasecmp makes the function POSIX but not ANSI
  compliant. When compling with Altivec, GCC Altivec extensions are
  supported.
*/

#define LOWE_COMPATIBLE

#include"mexutils.c"
#include<stdlib.h>
#include<math.h>

#ifdef WINDOWS
#include<string.h>
#ifndef __cplusplus
#define sqrtf(x)    ((float)sqrt((double)(x)))
#define powf(x,y)   ((float)pow((double)(x),(double)(y)))
#define fabsf(x)    ((float)fabs((double)(x)))
#define sinf(x)     ((float)sin((double)(x)))
#define cosf(x)     ((float)cos((double)(x)))
#define expf(x)     ((float)exp((double)(x)))
#define atan2f(x,y) ((float)atan2((double)(x),(double)(y)))
#endif
#else
#include<strings.h>
#endif

/* Altivec and Accelerate support.
 * Very crude at this time.
 */
#if defined( MACOSX ) && defined( __ALTIVEC__ )
#include<Accelerate/Accelerate.h>
typedef union 
{
  float x[4] ;
  vFloat vec ;
} float4 ;
#endif

#define greater(a,b) a > b
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


enum {SCALESPACE, NOSCALESPACE} ;

enum {PROP_MAGNIF=0,
      PROP_NBP,
      PROP_NBO,
      PROP_UNKNOWN} ;

char const * properties [4] = 
  { "Magnif",
    "NumSpatialBins",
    "NumOrientBins",
     0L
  } ;

/** Fast fmodf for 2*PI
 **/
/*inline*/
float fast_mod(float th)
{
  while(th < 0) th += 2*M_PI ;
  while(th > 2*M_PI) th -= 2*M_PI ;
  return th ;
}

/** Fast floor. Equivalent to (int) floor(x)
 **/
/*inline*/
int fast_floor(float x)
{
  return (int)( x - ((x>=0)?0:1) ) ; 
}

/** Normalizes in norm L_2 a descriptor.
 **/
void
normalize_histogram(float* L_begin, float* L_end)
{
  float* L_iter ;
  float norm=0.0 ;

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
    norm += (*L_iter) * (*L_iter) ;

  norm = sqrtf(norm) ;
  /*  mexPrintf("%f\n",norm) ;*/

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
    *L_iter /= norm ;
}

/** @brief MATLAB Driver.
 **/
void
mexFunction(int nout, mxArray *out[], 
            int nin, const mxArray *in[])
{
  int M,N,S=0,smin=0,K,num_levels=0 ;
  const int* dimensions ;
  const double* P_pt ;
  const double* G_pt ;
  float* descr_pt ;
  float* buffer_pt ;
  float sigma0 ;
  float magnif = 3.0f ; /* Spatial bin extension factor. */
  int NBP = 4 ;         /* Number of bins for one spatial direction (even). */
  int NBO = 8 ;         /* Number of bins for the ortientation. */
  int mode = NOSCALESPACE ;
  int buffer_size=0;

  enum {IN_G=0,IN_P,IN_SIGMA0,IN_S,IN_SMIN} ;
  enum {OUT_L=0} ;

  /* ------------------------------------------------------------------
  **                                                Check the arguments
  ** --------------------------------------------------------------- */ 
 
  if (nin < 3) {
    mexErrMsgTxt("At least three arguments are required") ;
  } else if (nout > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }
		
  if( !uIsRealScalar(in[IN_SIGMA0]) ) {
    mexErrMsgTxt("SIGMA0 should be a real scalar") ;
  }
	
  if(!mxIsDouble(in[IN_G]) ||
     mxGetNumberOfDimensions(in[IN_G]) > 3) {
    mexErrMsgTxt("G should be a real matrix or 3-D array") ;
  }
  
  sigma0 = (float) *mxGetPr(in[IN_SIGMA0]) ;
  
  dimensions = mxGetDimensions(in[IN_G]) ;
  M = dimensions[0] ;
  N = dimensions[1] ;
  G_pt = mxGetPr(in[IN_G]) ;
  
  P_pt = mxGetPr(in[IN_P]) ;	
  K = mxGetN(in[IN_P]) ;
  
  if( !uIsRealMatrix(in[IN_P],-1,-1)) {
    mexErrMsgTxt("P should be a real matrix") ;
  }

  if ( mxGetM(in[IN_P])  == 4) {
    /* Standard (scale space) mode */ 
    mode = SCALESPACE ;
    num_levels = dimensions[2] ;
    
    if(nin < 5) {
      mexErrMsgTxt("Five arguments are required in standard mode") ;
    }
    
    if( !uIsRealScalar(in[IN_S]) ) {
      mexErrMsgTxt("S should be a real scalar") ;
    }
    
    if( !uIsRealScalar(in[IN_SMIN]) ) {
      mexErrMsgTxt("SMIN should be a real scalar") ;
    }
    
    if( !uIsRealMatrix(in[IN_P],4,-1)) {
      mexErrMsgTxt("When the e mode P should be a 4xK matrix.") ;
    }
    
    S = (int)(*mxGetPr(in[IN_S])) ;
    smin = (int)(*mxGetPr(in[IN_SMIN])) ;
    
  } else if (  mxGetM(in[IN_P])  == 3 ) {
    mode = NOSCALESPACE ;
    num_levels = 1 ;
    S      = 1 ;
    smin   = 0 ;
  } else {
    mexErrMsgTxt("P should be either a 3xK or a 4xK matrix.") ;
  }

  /* Parse the property-value pairs */
  {
    char str [80] ;
    int arg = (mode == SCALESPACE) ? IN_SMIN + 1 : IN_SIGMA0 + 1 ;

    while(arg < nin) {
      int k ;

      if( !uIsString(in[arg],-1) ) {
        mexErrMsgTxt("The first argument in a property-value pair"
                     " should be a string") ;
      }
      mxGetString(in[arg], str, 80) ;

#ifdef WINDOWS      
      for(k = 0 ; properties[k] && strcmpi(str, properties[k]) ; ++k) ;
#else
      for(k = 0 ; properties[k] && strcasecmp(str, properties[k]) ; ++k) ;
#endif

      switch (k) {
      case PROP_NBP:
        if( !uIsRealScalar(in[arg+1]) ) {
          mexErrMsgTxt("'NumSpatialBins' should be real scalar") ;
        }
        NBP = (int) *mxGetPr(in[arg+1]) ;
        if( NBP <= 0 || (NBP & 0x1) ) {
          mexErrMsgTxt("'NumSpatialBins' must be positive and even") ;
        }
        break ;

      case PROP_NBO:
        if( !uIsRealScalar(in[arg+1]) ) {
          mexErrMsgTxt("'NumOrientBins' should be a real scalar") ;
        }
        NBO = (int) *mxGetPr(in[arg+1]) ;
        if( NBO <= 0 ) {
          mexErrMsgTxt("'NumOrientlBins' must be positive") ;
        }
        break ;

      case PROP_MAGNIF:
        if( !uIsRealScalar(in[arg+1]) ) {
          mexErrMsgTxt("'Magnif' should be a real scalar") ;
        }
        magnif = (float) *mxGetPr(in[arg+1]) ;
        if( magnif <= 0 ) {
          mexErrMsgTxt("'Magnif' must be positive") ;
        }
        break ;

      case PROP_UNKNOWN:
        mexErrMsgTxt("Property unknown.") ;
        break ;
      }
      arg += 2 ;
    }
  }
  
  /* -----------------------------------------------------------------
   *                                   Pre-compute gradient and angles
   * -------------------------------------------------------------- */
  /* Alloc two buffers and make sure their size is multiple of 128 for
   * better alignment (used also by the Altivec code below.)
   */
  buffer_size = (M*N*num_levels + 0x7f) & (~ 0x7f) ;
  buffer_pt = (float*) mxMalloc( sizeof(float) * 2 * buffer_size ) ;
  descr_pt  = (float*) mxCalloc( NBP*NBP*NBO*K,  sizeof(float)  ) ;

  {
    /* Offsets to move in the scale space. */
    const int yo = 1 ;
    const int xo = M ;
    const int so = M*N ;
    int x,y,s ;

#define at(x,y) (*(pt + (x)*xo + (y)*yo))

    /* Compute the gradient */
    for(s = 0 ; s < num_levels ; ++s) {
      const double* pt = G_pt + s*so ;
      for(x = 1 ; x < N-1 ; ++x) {
        for(y = 1 ; y < M-1 ; ++y) {
          float Dx = 0.5 * ( at(x+1,y) - at(x-1,y) ) ;
          float Dy = 0.5 * ( at(x,y+1) - at(x,y-1) ) ;
          buffer_pt[(x*xo+y*yo+s*so) + 0          ] = Dx ;
          buffer_pt[(x*xo+y*yo+s*so) + buffer_size] = Dy ;
        }
      }
    }
    
    /* Compute angles and modules */
    {
      float* pt = buffer_pt ;
      int j = 0 ;
      while (j < N*M*num_levels) {

#if defined( MACOSX ) && defined( __ALTIVEC__ )
        if( ((unsigned int)pt & 0x7) == 0 && j+3 < N*M*num_levels ) {
          /* If aligned to 128 bit and there are at least 4 pixels left */
          float4 a, b, c, d ;
          a.vec = vec_ld(0,(vector float*)(pt              )) ;
          b.vec = vec_ld(0,(vector float*)(pt + buffer_size)) ;
          c.vec = vatan2f(b.vec,a.vec) ;
          a.x[0] = a.x[0]*a.x[0]+b.x[0]*b.x[0] ;
          a.x[1] = a.x[1]*a.x[1]+b.x[1]*b.x[1] ;
          a.x[2] = a.x[2]*a.x[2]+b.x[2]*b.x[2] ;
          a.x[3] = a.x[3]*a.x[3]+b.x[3]*b.x[3] ;
          d.vec = vsqrtf(a.vec) ;
          vec_st(c.vec,0,(vector float*)(pt + buffer_size)) ;
          vec_st(d.vec,0,(vector float*)(pt              )) ;
          j += 4 ;
          pt += 4 ;
        } else {
#endif
          float Dx = *(pt              ) ;
          float Dy = *(pt + buffer_size) ;
          *(pt              ) = sqrtf(Dx*Dx + Dy*Dy) ;
          *(pt + buffer_size) = atan2f(Dy, Dx) ;
          j += 1 ;
          pt += 1 ;
#if defined( MACOSX ) && defined( __ALTIVEC__ )
        }
#endif

      }
    }
  }

  /* -----------------------------------------------------------------
   *                                                        Do the job
   * -------------------------------------------------------------- */ 
  if(K > 0) {    
    int p ;

    /* Offsets to move in the buffer */
    const int yo = 1 ;
    const int xo = M ;
    const int so = M*N ;

    /* Offsets to move in the descriptor. */
    /* Use Lowe's convention. */
    const int binto = 1 ;
    const int binyo = NBO * NBP ;
    const int binxo = NBO ;
    const int bino  = NBO * NBP * NBP ;

    for(p = 0 ; p < K ; ++p, descr_pt += bino) {
      /* The SIFT descriptor is a  three dimensional histogram of the position
       * and orientation of the gradient.  There are NBP bins for each spatial
       * dimesions and NBO  bins for the orientation dimesion,  for a total of
       * NBP x NBP x NBO bins.
       *
       * The support  of each  spatial bin  has an extension  of SBP  = 3sigma
       * pixels, where sigma is the scale  of the keypoint.  Thus all the bins
       * together have a  support SBP x NBP pixels wide  . Since weighting and
       * interpolation of  pixel is used, another  half bin is  needed at both
       * ends of  the extension. Therefore, we  need a square window  of SBP x
       * (NBP + 1) pixels. Finally, since the patch can be arbitrarly rotated,
       * we need to consider  a window 2W += sqrt(2) x SBP  x (NBP + 1) pixels
       * wide.
       */      
      const float x = (float) *P_pt++ ;
      const float y = (float) *P_pt++ ;
      const float s = (float) (mode == SCALESPACE) ? (*P_pt++) : 0.0 ;
      const float theta0 = (float) *P_pt++ ;

      const float st0 = sinf(theta0) ;
      const float ct0 = cosf(theta0) ;
      const int xi = (int) floor(x+0.5) ; /* Round-off */
      const int yi = (int) floor(y+0.5) ;
      const int si = (int) floor(s+0.5) - smin ;
      const float sigma = sigma0 * powf(2, s / S) ;
      const float SBP = magnif * sigma ;
      const int W = (int) floor( sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;      
      int bin ;
      int dxi ;
      int dyi ;
      const float* pt ;
      float* dpt ;

      /* Check that keypoints are within bounds . */

      if(xi < 0   || 
         xi > N-1 || 
         yi < 0   || 
         yi > M-1 ||
         ((mode==SCALESPACE) && 
          (si < 0   ||
           si > dimensions[2]-1) ) )
        continue ;

      /* Center the scale space and the descriptor on the current keypoint. 
       * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
       */
      pt  = buffer_pt + xi*xo + yi*yo + si*so ;
      dpt = descr_pt + (NBP/2) * binyo + (NBP/2) * binxo ;
     
#define atd(dbinx,dbiny,dbint) (*(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo))
      
      /*
       * Process each pixel in the window and in the (1,1)-(M-1,N-1)
       * rectangle.
       */
      for(dxi = max(-W, 1-xi) ; dxi <= min(+W, N-2-xi) ; ++dxi) {
        for(dyi = max(-W, 1-yi) ; dyi <= min(+W, M-2-yi) ; ++dyi) {

          /* Compute the gradient. */
          float mod   = *(pt + dxi*xo + dyi*yo + 0           ) ;
          float angle = *(pt + dxi*xo + dyi*yo + buffer_size ) ;
#ifdef LOWE_COMPATIBLE
          float theta = fast_mod(-angle + theta0) ;
#else
          float theta = fast_mod(angle - theta0) ;
#endif
          /* Get the fractional displacement. */
          float dx = ((float)(xi+dxi)) - x;
          float dy = ((float)(yi+dyi)) - y;

          /* Get the displacement normalized w.r.t. the keypoint orientation
           * and extension. */
          float nx = ( ct0 * dx + st0 * dy) / SBP ;
          float ny = (-st0 * dx + ct0 * dy) / SBP ; 
          float nt = NBO * theta / (2*M_PI) ;

          /* Get the gaussian weight of the sample. The gaussian window
           * has a standard deviation of NBP/2. Note that dx and dy are in
           * the normalized frame, so that -NBP/2 <= dx <= NBP/2. */
          const float wsigma = NBP/2 ;
          float win =  expf(-(nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;

          /* The sample will be distributed in 8 adijacient bins. 
           * Now we get the ``lower-left'' bin. */
          int binx = fast_floor( nx - 0.5 ) ;
          int biny = fast_floor( ny - 0.5 ) ;
          int bint = fast_floor( nt ) ;
          float rbinx = nx - (binx+0.5) ;
          float rbiny = ny - (biny+0.5) ;
          float rbint = nt - bint ;
          int dbinx ;
          int dbiny ;
          int dbint ;

          /* Distribute the current sample into the 8 adijacient bins. */
          for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
            for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
              for(dbint = 0 ; dbint < 2 ; ++dbint) {
                
                if( binx+dbinx >= -(NBP/2) &&
                    binx+dbinx <   (NBP/2) &&
                    biny+dbiny >= -(NBP/2) &&
                    biny+dbiny <   (NBP/2) ) {
                  float weight = win 
                    * mod 
                    * fabsf(1 - dbinx - rbinx)
                    * fabsf(1 - dbiny - rbiny)
                    * fabsf(1 - dbint - rbint) ;

                  atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
                }
              }            
            }
          }
        }  
      }

      {
        /* Normalize the histogram to L2 unit length. */        
        normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;
        
        /* Truncate at 0.2. */
        for(bin = 0; bin < NBO*NBP*NBP ; ++bin) {
          if (descr_pt[bin] > 0.2) descr_pt[bin] = 0.2;
        }
        
        /* Normalize again. */
        normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;
      }
    }
  }

  /* Restore pointer to the beginning of the descriptors. */
  descr_pt -= NBO*NBP*NBP*K ;

  {
    int k ;
    double* L_pt ;
    out[OUT_L] = mxCreateDoubleMatrix(NBP*NBP*NBO, K, mxREAL) ;
    L_pt = mxGetPr(out[OUT_L]) ;
    for(k = 0 ; k < NBP*NBP*NBO*K ; ++k) {
      L_pt[k] = descr_pt[k] ;
    }
  }

  mxFree(descr_pt) ;  
  mxFree(buffer_pt) ;
}
