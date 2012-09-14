/* file:        siftrefinemx.c
** author:      Andrea Vedaldi
** description: Subpixel localization, thresholding and off-edge test.
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

#include"mex.h"

#include<mexutils.c>

#include<stdlib.h>
#include<string.h>

#ifdef WINDOWS
#undef min
#undef max
#ifndef __cplusplus__
#define sqrtf(x) ((float)sqrt((double)x)
#define powf(x)  ((float)pow((double)x)
#define fabsf(x) ((float)fabs((double)x)
#endif
#endif

#define greater(a,b) ((a) > (b))
#define min(a,b)     (((a)<(b))?(a):(b))
#define max(a,b)     (((a)>(b))?(a):(b))
#define abs(a)       (((a)>0)?(a):(-a))

const int max_iter = 5 ;

void
mexFunction(int nout, mxArray *out[],
            int nin, const mxArray *in[])
{
  int M,N,S,smin,K ;
  const int* dimensions ;
  const double* P_pt ;
  const double* D_pt ;
  double threshold = 0.01 ; /*0.02 ;*/
  double r = 10.0 ;
  double* result ;
  enum {IN_P=0,IN_D,IN_SMIN,IN_THRESHOLD,IN_R} ;
  enum {OUT_Q=0} ;

  /* -----------------------------------------------------------------
  **                                               Check the arguments
  ** -------------------------------------------------------------- */
  if (nin < 3) {
    mexErrMsgTxt("At least three input arguments required.");
  } else if (nout > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  if( !uIsRealMatrix(in[IN_P],3,-1) ) {
    mexErrMsgTxt("P must be a 3xK real matrix") ;
  }

  if( !mxIsDouble(in[IN_D]) || mxGetNumberOfDimensions(in[IN_D]) != 3) {
    mexErrMsgTxt("G must be a three dimensional real array.") ;
  }

  if( !uIsRealScalar(in[IN_SMIN]) ) {
    mexErrMsgTxt("SMIN must be a real scalar.") ;
  }

  if(nin >= 4) {
    if(!uIsRealScalar(in[IN_THRESHOLD])) {
      mexErrMsgTxt("THRESHOLD must be a real scalar.") ;
    }
    threshold = *mxGetPr(in[IN_THRESHOLD]) ;
  }

  if(nin >= 5) {
    if(!uIsRealScalar(in[IN_R])) {
      mexErrMsgTxt("R must be a real scalar.") ;
    }
    r = *mxGetPr(in[IN_R]) ;
  }

  dimensions = mxGetDimensions(in[IN_D]) ;
  M = dimensions[0] ;
  N = dimensions[1] ;
  S = dimensions[2] ;
  smin = (int)(*mxGetPr(in[IN_SMIN])) ;

  if(S < 3 || M < 3 || N < 3) {
    mexErrMsgTxt("All dimensions of DOG must be not less than 3.") ;
  }

  K = mxGetN(in[IN_P]) ;
  P_pt = mxGetPr(in[IN_P]) ;
  D_pt = mxGetPr(in[IN_D]) ;

  /* If the input array is empty, then output an empty array as well. */
  if( K == 0) {
    out[OUT_Q] = mxDuplicateArray(in[IN_P]) ;
    return ;
  }

  /* -----------------------------------------------------------------
   *                                                        Do the job
   * -------------------------------------------------------------- */
  {
    double* buffer = (double*) mxMalloc(K*3*sizeof(double)) ;
    double* buffer_iterator = buffer ;
    int p ;
    const int yo = 1 ;
    const int xo = M ;
    const int so = M*N ;

    for(p = 0 ; p < K ; ++p) {
      int x = ((int)*P_pt++) ;
      int y = ((int)*P_pt++) ;
      int s = ((int)*P_pt++) - smin ;
      int iter ;
      double b[3] ;

      /* Local maxima extracted from the DOG
       * have coorrinates 1<=x<=N-2, 1<=y<=M-2
       * and 1<=s-mins<=S-2. This is also the range of the points
       * that we can refine.
       */
      if(x < 1 || x > N-2 ||
         y < 1 || y > M-2 ||
         s < 1 || s > S-2) {
        continue ;
      }

#define at(dx,dy,ds) (*(pt + (dx)*xo + (dy)*yo + (ds)*so))

      {
        const double* pt = D_pt + y*yo + x*xo + s*so ;
        double Dx=0,Dy=0,Ds=0,Dxx=0,Dyy=0,Dss=0,Dxy=0,Dxs=0,Dys=0 ;
        int dx = 0 ;
        int dy = 0 ;
        int j, i, jj, ii ;

        for(iter = 0 ; iter < max_iter ; ++iter) {

          double A[3*3] ;

#define Aat(i,j) (A[(i)+(j)*3])

          x += dx ;
          y += dy ;
          pt = D_pt + y*yo + x*xo + s*so ;

          /* Compute the gradient. */
          Dx = 0.5 * (at(+1,0,0) - at(-1,0,0)) ;
          Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
          Ds = 0.5 * (at(0,0,+1) - at(0,0,-1)) ;

          /* Compute the Hessian. */
          Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0)) ;
          Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0)) ;
          Dss = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0)) ;

          Dxy = 0.25 * ( at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0) ) ;
          Dxs = 0.25 * ( at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1) ) ;
          Dys = 0.25 * ( at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1) ) ;

          /* Solve linear system. */
          Aat(0,0) = Dxx ;
          Aat(1,1) = Dyy ;
          Aat(2,2) = Dss ;
          Aat(0,1) = Aat(1,0) = Dxy ;
          Aat(0,2) = Aat(2,0) = Dxs ;
          Aat(1,2) = Aat(2,1) = Dys ;

          b[0] = - Dx ;
          b[1] = - Dy ;
          b[2] = - Ds ;

          /* Gauss elimination */
          for(j = 0 ; j < 3 ; ++j) {
            double maxa    = 0 ;
            double maxabsa = 0 ;
            int    maxi    = -1 ;
            double tmp ;

            /* look for the maximally stable pivot */
            for (i = j ; i < 3 ; ++i) {
              double a    = Aat (i,j) ;
              double absa = abs (a) ;
              if (absa > maxabsa) {
                maxa    = a ;
                maxabsa = absa ;
                maxi    = i ;
              }
            }

            /* if singular give up */
            if (maxabsa < 1e-10f) {
              b[0] = 0 ;
              b[1] = 0 ;
              b[2] = 0 ;
              break ;
            }

            i = maxi ;

            /* swap j-th row with i-th row and normalize j-th row */
            for(jj = j ; jj < 3 ; ++jj) {
              tmp = Aat(i,jj) ; Aat(i,jj) = Aat(j,jj) ; Aat(j,jj) = tmp ;
              Aat(j,jj) /= maxa ;
            }
            tmp = b[j] ; b[j] = b[i] ; b[i] = tmp ;
            b[j] /= maxa ;

            /* elimination */
            for (ii = j+1 ; ii < 3 ; ++ii) {
              double x = Aat(ii,j) ;
              for (jj = j ; jj < 3 ; ++jj) {
                Aat(ii,jj) -= x * Aat(j,jj) ;
              }
              b[ii] -= x * b[j] ;
            }
          }

          /* backward substitution */
          for (i = 2 ; i > 0 ; --i) {
            double x = b[i] ;
            for (ii = i-1 ; ii >= 0 ; --ii) {
              b[ii] -= x * Aat(ii,i) ;
            }
          }

          /* If the translation of the keypoint is big, move the keypoint
           * and re-iterate the computation. Otherwise we are all set.
           */
          dx= ((b[0] >  0.6 && x < N-2) ?  1 : 0 )
            + ((b[0] < -0.6 && x > 1  ) ? -1 : 0 ) ;

          dy= ((b[1] >  0.6 && y < M-2) ?  1 : 0 )
            + ((b[1] < -0.6 && y > 1  ) ? -1 : 0 ) ;

          if( dx == 0 && dy == 0 ) break ;

        }

        {
          double val = at(0,0,0) + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]) ;
          double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ;
          double xn = x + b[0] ;
          double yn = y + b[1] ;
          double sn = s + b[2] ;

          if(fabs(val) > threshold &&
             score < (r+1)*(r+1)/r &&
             score >= 0 &&
             fabs(b[0]) < 1.5 &&
             fabs(b[1]) < 1.5 &&
             fabs(b[2]) < 1.5 &&
             xn >= 0 &&
             xn <= N-1 &&
             yn >= 0 &&
             yn <= M-1 &&
             sn >= 0 &&
             sn <= S-1) {
            *buffer_iterator++ = xn ;
            *buffer_iterator++ = yn ;
            *buffer_iterator++ = sn+smin  ;
          }
        }
      }
    }

    /* Copy the result into an array. */
    {
      int NL = (buffer_iterator - buffer)/3 ;
      out[OUT_Q] = mxCreateDoubleMatrix(3, NL, mxREAL) ;
      result = mxGetPr(out[OUT_Q]);
      memcpy(result, buffer, sizeof(double) * 3 * NL) ;
    }
    mxFree(buffer) ;
  }

}


