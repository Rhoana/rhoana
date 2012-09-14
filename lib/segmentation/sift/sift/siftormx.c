/* file:        siftormx.c
** author:      Andrea Vedaldi
** description: Computes peaks of orientation histogram.
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
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<assert.h>

#include<mexutils.c>

#define greater(a,b)  ((a)>(b))
#define min(a,b)     (((a)<(b))?(a):(b))
#define max(a,b)     (((a)>(b))?(a):(b))

const double win_factor = 1.5 ;
#define NBINS 36
/* #define LOWE_BUG */

void
mexFunction(int nout, mxArray *out[],
            int nin, const mxArray *in[])
{
  int M,N,S,smin,K ;
  const int* dimensions ;
  const double* P_pt ;
  const double* G_pt ;
  double* TH_pt ;
  double sigma0 ;
  double H_pt [ NBINS ] ;

  enum {IN_P=0,IN_G,IN_S,IN_SMIN,IN_SIGMA0} ;
  enum {OUT_Q=0} ;

  /* -----------------------------------------------------------------
  **                                               Check the arguments
  ** -------------------------------------------------------------- */
  if (nin != 5) {
    mexErrMsgTxt("Exactly five input arguments required.");
  } else if (nout > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  if( !uIsRealScalar(in[IN_S]) ) {
    mexErrMsgTxt("S should be a real scalar") ;
  }

  if( !uIsRealScalar(in[IN_SMIN]) ) {
    mexErrMsgTxt("SMIN should be a real scalar") ;
  }

  if( !uIsRealScalar(in[IN_SIGMA0]) ) {
    mexErrMsgTxt("SIGMA0 should be a real scalar") ;
  }

  if( !uIsRealMatrix(in[IN_P],3,-1)) {
    mexErrMsgTxt("P should be a 3xK real matrix") ;
  }

  if(mxGetNumberOfDimensions(in[IN_G]) != 3) {
    mexErrMsgTxt("SSO must be a three dimensional array") ;
  }

  dimensions = mxGetDimensions(in[IN_G]) ;
  M = dimensions[0] ;
  N = dimensions[1] ;
  S = (int)(*mxGetPr(in[IN_S])) ;
  smin = (int)(*mxGetPr(in[IN_SMIN])) ;
  sigma0 = *mxGetPr(in[IN_SIGMA0]) ;

  K = mxGetN(in[IN_P]) ;
  P_pt = mxGetPr(in[IN_P]) ;
  G_pt = mxGetPr(in[IN_G]) ;


  /* If the input array is empty, then output an empty array as well. */
  if(K == 0) {
    out[OUT_Q] = mxCreateDoubleMatrix(4,0,mxREAL) ;
    return ;
  }

  /* ------------------------------------------------------------------
   *                                                         Do the job
   * --------------------------------------------------------------- */
  {
    int p ;
    const int yo = 1 ;
    const int xo = M ;
    const int so = M*N ;

    int buffer_size = K*4 ;
    double* buffer_start = (double*) mxMalloc( buffer_size *sizeof(double)) ;
    double* buffer_iterator = buffer_start ;
    double* buffer_end = buffer_start + buffer_size ;

    for(p = 0 ; p < K ; ++p, TH_pt += 2) {
      const double x = *P_pt++ ;
      const double y = *P_pt++ ;
      const double s = *P_pt++ ;
      int xi = ((int) (x+0.5)) ; /* Round them off. */
      int yi = ((int) (y+0.5)) ;
      int si = ((int) (s+0.5)) - smin ;
      int xs ;
      int ys ;
      double sigmaw = win_factor * sigma0 * pow(2, ((double)s) / S) ;
      int W = (int) floor(3.0 * sigmaw) ;
      int bin ;
      const double* pt ;

      /* Make sure that the rounded off keypoint index is within bound.
       */
      if(xi < 0   ||
         xi > N-1 ||
         yi < 0   ||
         yi > M-1 ||
         si < 0   ||
         si > dimensions[2]-1 ) {
        mexPrintf("Dropping %d: W %d x %d y %d si [%d,%d,%d,%d]\n",p,W,xi,yi,si,M,N,dimensions[2]) ;
        continue ;
      }

      /* Clear histogram buffer. */
      {
        int i ;
        for(i = 0 ; i < NBINS ; ++i)
          H_pt[i] = 0 ;
      }

      pt = G_pt + xi*xo + yi*yo + si*so ;

#define at(dx,dy) (*(pt + (dx)*xo + (dy)*yo))

      for(xs = max(-W, 1-xi) ; xs <= min(+W, N -2 -xi) ; ++xs) {
        for(ys = max(-W, 1-yi) ; ys <= min(+W, M -2 -yi) ; ++ys) {
          double Dx = 0.5 * ( at(xs+1,ys) - at(xs-1,ys) ) ;
          double Dy = 0.5 * ( at(xs,ys+1) - at(xs,ys-1) ) ;
          double dx = ((double)(xi+xs)) - x;
          double dy = ((double)(yi+ys)) - y;

          if(dx*dx + dy*dy >= W*W+0.5) continue ;

          {
            double win = exp( - (dx*dx + dy*dy)/(2*sigmaw*sigmaw) ) ;
            double mod = sqrt(Dx*Dx + Dy*Dy) ;
            double theta = fmod(atan2(Dy, Dx) + 2*M_PI, 2*M_PI) ;
            bin = (int)( NBINS * theta / (2*M_PI) ) ;
            H_pt[bin] += mod*win ;
          }
        }
      }

      /* Smooth histogram */
      {
        int iter, i ;
#ifdef LOWE_BUG
        for (iter = 0; iter < 6; iter++) {
          double prev  = H_pt[NBINS/2] ;
          for (i = NBINS/2-1; i >= -NBINS/2 ; --i) {
            int j  = (i     + NBINS) % NBINS ;
            int jp = (i - 1 + NBINS) % NBINS ;
            double newh = (prev + H_pt[j] + H_pt[jp]) / 3.0;
            prev = H_pt[j] ;
            H_pt[j] = newh ;
          }
        }
#else
        for (iter = 0; iter < 6; iter++) {
          double prev;
          prev = H_pt[NBINS-1];
          for (i = 0; i < NBINS; i++) {
            double newh = (prev + H_pt[i] + H_pt[(i+1) % NBINS]) / 3.0;
            prev = H_pt[i] ;
            H_pt[i] = newh ;
          }
        }
#endif
      }

      /* Find strongest peaks. */
      {
        int i ;
        double maxh = H_pt[0] ;
        for(i = 1 ; i < NBINS ; ++i)
          maxh = max(maxh, H_pt[i]) ;

        for(i = 0 ; i < NBINS ; ++i) {
          double h0 = H_pt[i] ;
          double hm = H_pt[(i-1+NBINS) % NBINS] ;
          double hp = H_pt[(i+1+NBINS) % NBINS] ;

          if( h0 > 0.8*maxh && h0 > hm && h0 > hp ) {

            double di = -0.5 * (hp-hm) / (hp+hm-2*h0) ; /*di=0;*/
            double th = 2*M_PI*(i+di+0.5)/NBINS ;

            if( buffer_iterator == buffer_end ) {
              int offset = buffer_iterator - buffer_start ;
              buffer_size += 4*max(1, K/16) ;
              buffer_start = (double*) mxRealloc(buffer_start,
                                                 buffer_size*sizeof(double)) ;
              buffer_end = buffer_start + buffer_size ;
              buffer_iterator = buffer_start + offset ;
            }

            *buffer_iterator++ = x ;
            *buffer_iterator++ = y ;
            *buffer_iterator++ = s ;
            *buffer_iterator++ = th ;
          }
        } /* Scan histogram */
      } /* Find peaks */
    }

    /* Save back the result. */
    {
      double* result ;
      int NL = (buffer_iterator - buffer_start)/4 ;
      out[OUT_Q] = mxCreateDoubleMatrix(4, NL, mxREAL) ;
      result  = mxGetPr(out[OUT_Q]);
      memcpy(result, buffer_start, sizeof(double) * 4 * NL) ;
    }
    mxFree(buffer_start) ;
  }
}
