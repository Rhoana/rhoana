/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _CVUTILS_H_
#define _CVUTILS_H_

#include "cvtypes.h"
#include "cverror.h"

#ifdef ICV_COMPILE_PROFILE_
   #define icvAlloc(size)  icvAllocEx(size, __FILE__, __LINE__)
   #define icvFree(ptr)    icvFreeEx((void**)(ptr),__FILE__, __LINE__)
#else
   #define icvAlloc(size)  icvAllocEx(size, 0, 0)
   #define icvFree(ptr)    icvFreeEx((void**)(ptr), 0, 0)
#endif

/****************************************************************************************\
  
  Generic implementation of Quick Sort algorithm.
  ----------------------------------------------
  It allows to define qsort function for any certain type and
  any certain precedence rule.
  To use it you should define comparsion macro which takes two arguments
  and returns non-zero if first argument should precede the second and zero else.

  Example:

  Suppose that the task is to sort points by ascending of y coordinates and if
  y's are equal x's should ascend.

    The code is:
    ------------------------------------------------------------------------------
           #define cmp_pts( pt1, pt2 ) \
               ((pt1).y < (pt2).y || (pt1).y == (pt2).y && (pt1).x < (pt2).x)

           [static] IPCV_IMPLEMENT_QSORT( icvSortPoints, CvPoint, cmp_pts )
    ------------------------------------------------------------------------------

    This code generates function void icvSortPoints( CvPoint* array, int length )
    which can be used further.

    aux is an additional parameter, which can be used when comparing elements.

\****************************************************************************************/

#define CV_IMPLEMENT2_QSORT( func_name, T, less_than, user_data_type )  \
void func_name( T* array, int length, user_data_type aux )              \
{                                                                       \
    const int bubble_level = 8;                                         \
                                                                        \
    struct                                                              \
    {                                                                   \
        int lb, ub;                                                     \
    }                                                                   \
    stack[48];                                                          \
                                                                        \
    int sp = 0;                                                         \
                                                                        \
    T   temp;                                                           \
    T   lb_val;                                                         \
                                                                        \
    aux = aux;                                                          \
                                                                        \
    stack[0].lb = 0;                                                    \
    stack[0].ub = length - 1;                                           \
                                                                        \
    while( sp >= 0 )                                                    \
    {                                                                   \
        int lb = stack[sp].lb;                                          \
        int ub = stack[sp--].ub;                                        \
                                                                        \
        for(;;)                                                         \
        {                                                               \
            int diff = ub - lb;                                         \
            if( diff < bubble_level )                                   \
            {                                                           \
                int i, j;                                               \
                T* arr = array + lb;                                    \
                                                                        \
                for( i = diff; i > 0; i-- )                             \
                {                                                       \
                    int f = 0;                                          \
                    for( j = 0; j < i; j++ )                            \
                        if( less_than( arr[j+1], arr[j] ))              \
                        {                                               \
                            temp = arr[j];                              \
                            arr[j] = arr[j+1];                          \
                            arr[j+1] = temp;                            \
                            f = 1;                                      \
                        }                                               \
                    if( !f ) break;                                     \
                }                                                       \
                break;                                                  \
            }                                                           \
            else                                                        \
            {                                                           \
                /* select pivot and exchange with 1st element */        \
                int  m = lb + (diff >> 1);                              \
                int  i = lb + 1, j = ub;                                \
                                                                        \
                lb_val = array[m];                                      \
                                                                        \
                array[m]  = array[lb];                                  \
                array[lb] = lb_val;                                     \
                                                                        \
                /* partition into two segments */                       \
                for(;;)                                                 \
                {                                                       \
                    for( ;i < j && less_than(array[i], lb_val); i++ );  \
                    for( ;j >= i && less_than(lb_val, array[j]); j-- ); \
                                                                        \
                    if( i >= j ) break;                                 \
                    temp = array[i];                                    \
                    array[i++] = array[j];                              \
                    array[j--] = temp;                                  \
                }                                                       \
                                                                        \
                /* pivot belongs in A[j] */                             \
                array[lb] = array[j];                                   \
                array[j]  = lb_val;                                     \
                                                                        \
                /* keep processing smallest segment, and stack largest*/\
                if( j - lb <= ub - j )                                  \
                {                                                       \
                    if( j + 1 < ub )                                    \
                    {                                                   \
                        stack[++sp].lb   = j + 1;                       \
                        stack[sp].ub = ub;                              \
                    }                                                   \
                    ub = j - 1;                                         \
                }                                                       \
                else                                                    \
                {                                                       \
                    if( j - 1 > lb)                                     \
                    {                                                   \
                        stack[++sp].lb = lb;                            \
                        stack[sp].ub = j - 1;                           \
                    }                                                   \
                    lb = j + 1;                                         \
                }                                                       \
            }                                                           \
        }                                                               \
    }                                                                   \
}

#define CV_IMPLEMENT_QSORT( func_name, T, less_than )  \
    CV_IMPLEMENT2_QSORT( func_name, T, less_than, int )


/****************************************************************************************\
*                             Mathematical functions                                     *
\****************************************************************************************/

/* calculates 1/sqrt(val) */
double icvInvSqrt64d( double val );

/* 
   Finds L1 norm between two blocks.
   !!!! function doesn't perform EMMS !!!!
*/
int  icvCmpBlocksL1_8u_C1( const uchar* vec1, const uchar* vec2, int len );

/* 
   Finds squared L2 norm between two blocks.
   !!!! function don't perform EMMS (except 32f flavor function) !!!!
*/
int64  icvCmpBlocksL2_8u_C1( const uchar* vec1, const uchar* vec2, int len );
int64  icvCmpBlocksL2_8s_C1( const char* vec1, const char* vec2, int len );
double icvCmpBlocksL2_32f_C1( const float* vec1, const float* vec2, int len );


int64  icvSumPixels_8u_C1( const uchar* vec, int len );
int64  icvSumPixels_8s_C1( const char* vec, int len );
double icvSumPixels_32f_C1( const float* vec, int len );

/* 
   Calculates cross correlation for two blocks.
   !!!! function doesn't perform EMMS !!!!
*/
int64  icvCrossCorr_8u_C1( const uchar* vec1, const uchar* vec2, int len );
int64  icvCrossCorr_8s_C1( const char* vec1, const char* vec2, int len );
double icvCrossCorr_32f_C1( const float* vec1, const float* vec2, int len );

CvStatus  icvProjectPoints( int             numPoints,
                            CvPoint3D64d*   objectPoints,
                            CvVect64d       rotVect,
                            CvVect64d       transVect,
                            CvVect64d       focalLength,
                            CvPoint2D64d    principalPoint,
                            CvVect64d       distortion,
                            CvPoint2D64d*   imagePoints,
                            CvVect64d       derivPointsRot,
                            CvVect64d       derivPointsTrans,
                            CvVect64d       derivPointsFocal,
                            CvVect64d       derivPointsPrincipal,
                            CvVect64d       derivPointsDistort);


float  icvFastArctan32f( float y, float x );

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))
#define  CV_FLT_TO_FIX(x,n)  cvRound((x)*(1<<(n)))

int icvInitLineIterator( const CvMat* mat, CvPoint pt1, CvPoint pt2,
                         CvLineIterator* iterator,
                         int connectivity, int left_to_right = 0 );

void icvSepConvSmall3_32f( float*  src, int src_step,
                           float*  dst, int dst_step,
                           CvSize src_size,
                           const float* kx, const float* ky,
                           float*  buffer );

typedef struct CvFastDiv
{
    unsigned delta, scale, divisor;
}
CvFastDiv;

#define CV_FAST_DIV_SHIFT 32

CV_INLINE CvFastDiv icvFastDiv( int divisor );
CV_INLINE CvFastDiv icvFastDiv( int divisor )
{
    CvFastDiv fastdiv;
    
    assert( divisor >= 1 );
    uint64 temp = ((uint64)1 << CV_FAST_DIV_SHIFT)/divisor;

    fastdiv.divisor = divisor;
    fastdiv.delta = (unsigned)(((temp & 1) ^ 1) + divisor - 1);
    fastdiv.scale = (unsigned)((temp + 1) >> 1);
    
    return fastdiv;
}

#define CV_FAST_DIV( x, fastdiv )  \
    ((int)(((int64)((x)*2 + (int)(fastdiv).delta))*(int)(fastdiv).scale>>CV_FAST_DIV_SHIFT))

#define CV_FAST_UDIV( x, fastdiv )  \
    ((int)(((uint64)((x)*2 + (fastdiv).delta))*(fastdiv).scale>>CV_FAST_DIV_SHIFT))

/* calculate binomial kernel */
void icvCalcKer( char *kernel, int order, int size, int datatype, int origin );

/* extract packed color */
void icvExtractColor( double color, int type, void* buffer );

/* initialize accumulation table */
void icvInitAccTable( CvFuncTable* tabfl, CvFuncTable* tabdb,
                      CvBigFuncTable* masktab );

/****************************************************************************************\
*                                 IEEE 754 helper functions                              *
\****************************************************************************************/

#ifdef  CV_CHECK_FOR_NANS
    #define CV_CHECK_NANS( arr ) cvCheckArray((arr))  
    #define CV_CHECK_NANS_32F( src, srcstep, size ) \
        assert(icvCheckArray_32f_C1R( (src), (srcstep), (size), 0, 0, 0 )>=0)

    #define CV_CHECK_NANS_64F( src, srcstep, size ) \
        assert(icvCheckArray_64f_C1R( (src), (srcstep), (size), 0, 0, 0 )>=0)
#else
    #define CV_CHECK_NANS( arr )
    #define CV_CHECK_NANS_32F( src, srcstep, size )
    #define CV_CHECK_NANS_64F( src, srcstep, size )
#endif

#define icvCheckVector_32f( src, len ) \
    CV_CHECK_NANS_32F( (src), 0, cvSize(1,(len))) 

#define icvCheckVector_64f( src, len ) \
    CV_CHECK_NANS_64F( (src), 0, cvSize(1,(len))) 

#define icvCheckVector_64d icvCheckVector_64f

#define icvCheckMatrix_32f( src, w, h ) \
    CV_CHECK_NANS_32F( (src), 0, cvSize(1,(w)*(h))) 

#define icvCheckMatrix_64f( src, len ) \
    CV_CHECK_NANS_64F( (src), 0, cvSize(1,(w)*(h))) 

#define icvCheckMatrix_64d icvCheckMatrix_64f

#endif /*_IPCVUTILS_H_*/

/* End of file. */
