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

#ifndef __IPCV_H_
#define __IPCV_H_

/****************************************************************************************\
*                               Allocation/Deallocation                                  *
\****************************************************************************************/

IPCVAPI( void*, icvAllocEx, ( size_t size, const char* file, int line  ))
IPCVAPI( CVStatus, icvFreeEx, ( void** ptr, const char* file, int line ))


/****************************************************************************************\
*                                      Copy/Set                                          *
\****************************************************************************************/

IPCVAPI( CvStatus, icvCopy_8u_C1R, ( const uchar* src, int src_step,
                                     uchar* dst, int dst_step, CvSize size ))

IPCVAPI( CvStatus, icvSet_8u_C1R, ( uchar* dst, int dst_step, CvSize size,
                                    const void* scalar, int pix_size ))

IPCVAPI( CvStatus, icvSetZero_8u_C1R, ( uchar* dst, int dst_step, CvSize size ))


#define IPCV_CVT_TO( flavor )                                               \
IPCVAPI( CvStatus, icvCvtTo_##flavor##_C1R, ( const void* src, int step1,   \
                    void* dst, int step, CvSize size, int param ))

IPCV_CVT_TO( 8u )
IPCV_CVT_TO( 8s )
IPCV_CVT_TO( 16s )
IPCV_CVT_TO( 32s )
IPCV_CVT_TO( 32f )
IPCV_CVT_TO( 64f )

#undef IPCV_CVT_TO

IPCVAPI( CvStatus, icvCvt_32f64f, ( const float* src, double* dst, int len ))
IPCVAPI( CvStatus, icvCvt_64f32f, ( const double* src, float* dst, int len ))

/* dst(idx) = src(idx)*a + b */
IPCVAPI( CvStatus, icvScale_32f, ( const float* src, float* dst,
                                   int len, float a, float b ))
IPCVAPI( CvStatus, icvScale_64f, ( const double* src, double* dst,
                                   int len, double a, double b ))

/****************************************************************************************\
*                                       Arithmetics                                      *
\****************************************************************************************/

#define IPCV_BIN_ARITHM_NON_MASK( name )                            \
IPCVAPI( CvStatus, icv##name##_8u_C1R,                              \
( const uchar* src1, int srcstep1, const uchar* src2, int srcstep2, \
  uchar* dst, int dststep, CvSize size ))                           \
IPCVAPI( CvStatus, icv##name##_8s_C1R,                              \
( const char* src1, int srcstep1, const char* src2, int srcstep2,   \
  char* dst, int dststep, CvSize size ))                            \
IPCVAPI( CvStatus, icv##name##_16s_C1R,                             \
( const short* src1, int srcstep1, const short* src2, int srcstep2, \
  short* dst, int dststep, CvSize size ))                           \
IPCVAPI( CvStatus, icv##name##_32s_C1R,                             \
( const int* src1, int srcstep1, const int* src2, int srcstep2,     \
  int* dst, int dststep, CvSize size ))                             \
IPCVAPI( CvStatus, icv##name##_32f_C1R,                             \
( const float* src1, int srcstep1, const float* src2, int srcstep2, \
  float* dst, int dststep, CvSize size ))                           \
IPCVAPI( CvStatus, icv##name##_64f_C1R,                             \
( const double* src1, int srcstep1, const double* src2, int srcstep2,\
  double* dst, int dststep, CvSize size ))


#define IPCV_UN_ARITHM_NON_MASK( name )                             \
IPCVAPI( CvStatus, icv##name##_8u_C1R,                              \
( const uchar* src, int srcstep, uchar* dst, int dststep,           \
  CvSize size, const int* scalar ))                                 \
IPCVAPI( CvStatus, icv##name##_8s_C1R,                              \
( const char* src, int srcstep, char* dst, int dststep,             \
  CvSize size, const int* scalar ))                                 \
IPCVAPI( CvStatus, icv##name##_16s_C1R,                             \
( const short* src, int srcstep, short* dst, int dststep,           \
  CvSize size, const int* scalar ))                                 \
IPCVAPI( CvStatus, icv##name##_32s_C1R,                             \
( const int* src, int srcstep, int* dst, int dststep,               \
  CvSize size, const int* scalar ))                                 \
IPCVAPI( CvStatus, icv##name##_32f_C1R,                             \
( const float* src, int srcstep, float* dst, int dststep,           \
  CvSize size, const float* scalar ))                               \
IPCVAPI( CvStatus, icv##name##_64f_C1R,                             \
( const double* src, int srcstep, double* dst, int dststep,         \
  CvSize size, const double* scalar ))


IPCV_BIN_ARITHM_NON_MASK( Add )
IPCV_BIN_ARITHM_NON_MASK( Sub )
IPCV_UN_ARITHM_NON_MASK( AddC )
IPCV_UN_ARITHM_NON_MASK( SubRC )

#undef IPCV_BIN_ARITHM_NON_MASK
#undef IPCV_UN_ARITHM_NON_MASK

#define IPCV_MUL( flavor, arrtype )                                 \
IPCVAPI( CvStatus, icvMul_##flavor##_C1R,                           \
( const arrtype* src1, int step1, const arrtype* src2, int step2,   \
  arrtype* dst, int step, CvSize size, double scale ))

IPCV_MUL( 8u, uchar )
IPCV_MUL( 8s, char )
IPCV_MUL( 16s, short )
IPCV_MUL( 32s, int )
IPCV_MUL( 32f, float )
IPCV_MUL( 64f, double )

#undef IPCV_MUL

/****************************************************************************************\
*                                     Logical operations                                 *
\****************************************************************************************/

#define IPCV_LOGIC( name )                                              \
IPCVAPI( CvStatus, icv##name##_8u_C1R,                                  \
( const uchar* src1, int srcstep1, const uchar* src2, int srcstep2,     \
  uchar* dst, int dststep, CvSize size ))                               \
IPCVAPI( CvStatus, icv##name##C_8u_C1R,                                 \
( const uchar* src1, int srcstep1, uchar* dst, int dststep,             \
  CvSize, const uchar* scalar, int pix_size ))

IPCV_LOGIC( And )
IPCV_LOGIC( Or )
IPCV_LOGIC( Xor )

#undef IPCV_LOGIC

IPCVAPI( CvStatus, icvNot_8u_C1R,
( const uchar* src, int step1, uchar* dst, int step, CvSize size ))

/****************************************************************************************\
*                                Image Statistics                                        *
\****************************************************************************************/

///////////////////////////////////////// Mean //////////////////////////////////////////

#define IPCV_DEF_MEAN_MASK( flavor, srctype )                                           \
IPCVAPI( CvStatus, icvMean_##flavor##_C1MR, ( const srctype* img, int imgstep,          \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean ))              \
IPCVAPI( CvStatus, icvMean_##flavor##_C2MR, ( const srctype* img, int imgstep,          \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean ))              \
IPCVAPI( CvStatus, icvMean_##flavor##_C3MR, ( const srctype* img, int imgstep,          \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean ))              \
IPCVAPI( CvStatus, icvMean_##flavor##_C4MR, ( const srctype* img, int imgstep,          \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean ))              \
IPCVAPI( CvStatus, icvMean_##flavor##_CnCMR, ( const srctype* img, int imgstep,         \
                                               const uchar* mask, int maskStep,         \
                                               CvSize size, int cn,                     \
                                               int coi, double* mean))

IPCV_DEF_MEAN_MASK( 8u, uchar )
IPCV_DEF_MEAN_MASK( 8s, char )
IPCV_DEF_MEAN_MASK( 16s, short )
IPCV_DEF_MEAN_MASK( 32s, int )
IPCV_DEF_MEAN_MASK( 32f, float )
IPCV_DEF_MEAN_MASK( 64f, double )

#undef IPCV_DEF_MEAN_MASK

//////////////////////////////////// Mean_StdDev ////////////////////////////////////////

#define IPCV_DEF_MEAN_SDV( flavor, srctype )                                            \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C1R,( const srctype* img, int imgstep,     \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C2R, ( const srctype* img, int imgstep,    \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C3R, ( const srctype* img, int imgstep,    \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C4R, ( const srctype* img, int imgstep,    \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_CnCR,( const srctype* img, int imgstep,    \
                                              CvSize size, int cn,                      \
                                              int coi, double* mean, double* sdv ))     \
                                                                                        \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C1MR, ( const srctype* img, int imgstep,   \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C2MR, ( const srctype* img, int imgstep,   \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C3MR, ( const srctype* img, int imgstep,   \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_C4MR, ( const srctype* img, int imgstep,   \
                                              const uchar* mask, int maskStep,          \
                                              CvSize size, double* mean, double* sdv )) \
IPCVAPI( CvStatus, icvMean_StdDev_##flavor##_CnCMR, ( const srctype* img, int imgstep,  \
                                               const uchar* mask, int maskStep,         \
                                               CvSize size, int cn,                     \
                                               int coi, double* mean, double* sdv ))

IPCV_DEF_MEAN_SDV( 8u, uchar )
IPCV_DEF_MEAN_SDV( 8s, char )
IPCV_DEF_MEAN_SDV( 16s, short )
IPCV_DEF_MEAN_SDV( 32s, int )
IPCV_DEF_MEAN_SDV( 32f, float )
IPCV_DEF_MEAN_SDV( 64f, double )

#undef IPCV_DEF_MEAN_SDV

//////////////////////////////////// MinMaxIndx /////////////////////////////////////////


#define IPCV_DEF_MIN_MAX_LOC( flavor, srctype, extrtype )                               \
IPCVAPI( CvStatus, icvMinMaxIndx_##flavor##_C1R,( const srctype* img, int imgstep,      \
                                       CvSize size, extrtype* minVal, extrtype* maxVal, \
                                       CvPoint* minLoc, CvPoint* maxLoc ))              \
IPCVAPI( CvStatus, icvMinMaxIndx_##flavor##_CnCR,( const srctype* img, int imgstep,     \
                                       CvSize size, int cn, int coi,                    \
                                       extrtype* minVal, extrtype* maxVal,              \
                                       CvPoint* minLoc, CvPoint* maxLoc ))              \
                                                                                        \
IPCVAPI( CvStatus, icvMinMaxIndx_##flavor##_C1MR, ( const srctype* img, int imgstep,    \
                                       const uchar* mask, int maskStep,                 \
                                       CvSize size, extrtype* minVal, extrtype* maxVal, \
                                       CvPoint* minLoc, CvPoint* maxLoc ))              \
                                                                                        \
IPCVAPI( CvStatus, icvMinMaxIndx_##flavor##_CnCMR, ( const srctype* img, int imgstep,   \
                                       const uchar* mask, int maskStep,                 \
                                       CvSize size, int cn, int coi,                    \
                                       extrtype* minVal, extrtype* maxVal,              \
                                       CvPoint* minLoc, CvPoint* maxLoc ))

IPCV_DEF_MIN_MAX_LOC( 8u, uchar, float )
IPCV_DEF_MIN_MAX_LOC( 8s, char, float )
IPCV_DEF_MIN_MAX_LOC( 16s, short, float )
IPCV_DEF_MIN_MAX_LOC( 32s, int, double )
IPCV_DEF_MIN_MAX_LOC( 32f, float, float )
IPCV_DEF_MIN_MAX_LOC( 64f, double, double )

#undef IPCV_MIN_MAX_LOC

////////////////////////////////////////// Sum //////////////////////////////////////////

#define IPCV_DEF_SUM( flavor, srctype )                                     \
IPCVAPI( CvStatus, icvSum_##flavor##_C1R,( const srctype* img, int imgstep, \
                                           CvSize size, double* sum ))      \
IPCVAPI( CvStatus, icvSum_##flavor##_C2R,( const srctype* img, int imgstep, \
                                           CvSize size, double* sum ))      \
IPCVAPI( CvStatus, icvSum_##flavor##_C3R,( const srctype* img, int imgstep, \
                                           CvSize size, double* sum ))      \
IPCVAPI( CvStatus, icvSum_##flavor##_C4R,( const srctype* img, int imgstep, \
                                           CvSize size, double* sum ))      \
IPCVAPI( CvStatus, icvSum_##flavor##_CnCR,( const srctype* img, int imgstep,\
                                            CvSize size, int cn,            \
                                            int coi, double* sum ))

IPCV_DEF_SUM( 8u, uchar )
IPCV_DEF_SUM( 8s, char )
IPCV_DEF_SUM( 16s, short )
IPCV_DEF_SUM( 32s, int )
IPCV_DEF_SUM( 32f, float )
IPCV_DEF_SUM( 64f, double )

#undef IPCV_DEF_SUM

////////////////////////////////////////// CountNonZero /////////////////////////////////

#define IPCV_DEF_NON_ZERO( flavor, srctype )                                            \
IPCVAPI( CvStatus, icvCountNonZero_##flavor##_C1R,( const srctype* img, int imgstep,    \
                                                    CvSize size, int* nonzero ))        \
IPCVAPI( CvStatus, icvCountNonZero_##flavor##_CnCR,( const srctype* img, int imgstep,   \
                                                     CvSize size, int cn,               \
                                                     int coi, int* nonzero ))

IPCV_DEF_NON_ZERO( 8u, uchar )
IPCV_DEF_NON_ZERO( 16s, ushort )
IPCV_DEF_NON_ZERO( 32s, int )
IPCV_DEF_NON_ZERO( 32f, int )
IPCV_DEF_NON_ZERO( 64f, int64 )

#undef IPCV_DEF_NON_ZERO

//////////////////////////////////////// Moments ////////////////////////////////////////

#define IPCV_DEF_MOMENTS( name, flavor, srctype )       \
IPCVAPI( CvStatus, icv##name##_##flavor##_CnCR,         \
( const srctype* img, int step, CvSize size, int cn, int coi, double *moments ))

IPCV_DEF_MOMENTS( MomentsInTile, 8u, uchar )
IPCV_DEF_MOMENTS( MomentsInTile, 8s, char )
IPCV_DEF_MOMENTS( MomentsInTile, 16s, short )
IPCV_DEF_MOMENTS( MomentsInTile, 32f, float )
IPCV_DEF_MOMENTS( MomentsInTile, 64f, double )

IPCV_DEF_MOMENTS( MomentsInTileBin, 8u, uchar )
IPCV_DEF_MOMENTS( MomentsInTileBin, 16s, ushort )
IPCV_DEF_MOMENTS( MomentsInTileBin, 32f, int )
IPCV_DEF_MOMENTS( MomentsInTileBin, 64f, int64 )

#undef IPCV_DEF_MOMENTS

////////////////////////////////////////// Norm 1 /////////////////////////////////


#define IPCV_DEF_NORM_C1( flavor, srctype )                                             \
IPCVAPI( CvStatus, icvNorm_Inf_##flavor##_C1R,( const srctype* img, int imgstep,        \
                                                CvSize size, double* norm ))            \
IPCVAPI( CvStatus, icvNorm_L1_##flavor##_C1R,( const srctype* img, int imgstep,         \
                                                CvSize size, double* norm ))            \
IPCVAPI( CvStatus, icvNorm_L2_##flavor##_C1R,( const srctype* img, int imgstep,         \
                                                CvSize size, double* norm ))            \
IPCVAPI( CvStatus, icvNormDiff_Inf_##flavor##_C1R,( const srctype* img1, int imgstep1,  \
                                                    const srctype* img2, int imgstep2,  \
                                                    CvSize size, double* norm ))        \
IPCVAPI( CvStatus, icvNormDiff_L1_##flavor##_C1R,( const srctype* img1, int imgstep1,   \
                                                   const srctype* img2, int imgstep2,   \
                                                   CvSize size, double* norm ))         \
IPCVAPI( CvStatus, icvNormDiff_L2_##flavor##_C1R,( const srctype* img1, int imgstep1,   \
                                                   const srctype* img2, int imgstep2,   \
                                                   CvSize size, double* norm ))         \
IPCVAPI( CvStatus, icvNorm_Inf_##flavor##_C1MR,( const srctype* img, int imgstep,       \
                                                 const uchar* mask, int maskstep,       \
                                                 CvSize size, double* norm ))           \
IPCVAPI( CvStatus, icvNorm_L1_##flavor##_C1MR,( const srctype* img, int imgstep,        \
                                                const uchar* mask, int maskstep,        \
                                                CvSize size, double* norm ))            \
IPCVAPI( CvStatus, icvNorm_L2_##flavor##_C1MR,( const srctype* img, int imgstep,        \
                                                const uchar* mask, int maskstep,        \
                                                CvSize size, double* norm ))            \
IPCVAPI( CvStatus, icvNormDiff_Inf_##flavor##_C1MR,( const srctype* img1, int imgstep1, \
                                                    const srctype* img2, int imgstep2,  \
                                                    const uchar* mask, int maskstep,    \
                                                    CvSize size, double* norm ))        \
IPCVAPI( CvStatus, icvNormDiff_L1_##flavor##_C1MR,( const srctype* img1, int imgstep1,  \
                                                    const srctype* img2, int imgstep2,  \
                                                    const uchar* mask, int maskstep,    \
                                                   CvSize size, double* norm ))         \
IPCVAPI( CvStatus, icvNormDiff_L2_##flavor##_C1MR,( const srctype* img1, int imgstep1,  \
                                                    const srctype* img2, int imgstep2,  \
                                                    const uchar* mask, int maskstep,    \
                                                   CvSize size, double* norm ))

IPCV_DEF_NORM_C1( 8u, uchar )
IPCV_DEF_NORM_C1( 8s, char )
IPCV_DEF_NORM_C1( 16s, short )
IPCV_DEF_NORM_C1( 32s, int )
IPCV_DEF_NORM_C1( 32f, float )
IPCV_DEF_NORM_C1( 64f, double )

#undef IPCV_DEF_NORM_C1

/****************************************************************************************\
*                                       Utilities                                        *
\****************************************************************************************/

////////////////////////////// Copy Pixel <-> Plane /////////////////////////////////

#define IPCV_PIX_PLANE( flavor, arrtype )                                           \
IPCVAPI( CvStatus, icvCopy_##flavor##_C2P2R,                                        \
    ( const arrtype* src, int srcstep, arrtype** dst, int dststep, CvSize size ))   \
IPCVAPI( CvStatus, icvCopy_##flavor##_C3P3R,                                        \
    ( const arrtype* src, int srcstep, arrtype** dst, int dststep, CvSize size ))   \
IPCVAPI( CvStatus, icvCopy_##flavor##_C4P4R,                                        \
    ( const arrtype* src, int srcstep, arrtype** dst, int dststep, CvSize size ))   \
IPCVAPI( CvStatus, icvCopy_##flavor##_CnC1CR,                                       \
    ( const arrtype* src, int srcstep, arrtype* dst, int dststep,                   \
      CvSize size, int cn, int coi ))                                               \
IPCVAPI( CvStatus, icvCopy_##flavor##_C1CnCR,                                       \
    ( const arrtype* src, int srcstep, arrtype* dst, int dststep,                   \
      CvSize size, int cn, int coi ))                                               \
IPCVAPI( CvStatus, icvCopy_##flavor##_P2C2R,                                        \
    ( const arrtype** src, int srcstep, arrtype* dst, int dststep, CvSize size ))   \
IPCVAPI( CvStatus, icvCopy_##flavor##_P3C3R,                                        \
    ( const arrtype** src, int srcstep, arrtype* dst, int dststep, CvSize size ))   \
IPCVAPI( CvStatus, icvCopy_##flavor##_P4C4R,                                        \
    ( const arrtype** src, int srcstep, arrtype* dst, int dststep, CvSize size ))

IPCV_PIX_PLANE( 8u, uchar )
IPCV_PIX_PLANE( 16u, ushort )
IPCV_PIX_PLANE( 32s, int )
IPCV_PIX_PLANE( 64f, int64 )

#undef IPCV_PIX_PLANE

/****************************************************************************************\
*                                  Background differencing                               *
\****************************************************************************************/

////////////////////////////////////// AbsDiff ///////////////////////////////////////////

#define IPCV_ABS_DIFF( flavor, arrtype, scalartype )                                \
IPCVAPI( CvStatus, icvAbsDiff_##flavor##_C1R,                                       \
    ( const arrtype* src1, int srcstep1, const arrtype* src2, int srcstep2,         \
      arrtype* dst, int dststep, CvSize size ))

IPCV_ABS_DIFF( 8u, uchar, int )
IPCV_ABS_DIFF( 32f, float, float )
IPCV_ABS_DIFF( 64f, double, double )

#undef IPCV_ABS_DIFF

/////////////////////////////////// Accumulation /////////////////////////////////////////

#define IPCV_ACCUM( flavor, arrtype, acctype )                                      \
IPCVAPI( CvStatus, icvAdd_##flavor##_C1IR,                                          \
    ( const arrtype* src, int srcstep, acctype* dst, int dststep, CvSize size ))    \
IPCVAPI( CvStatus, icvAddSquare_##flavor##_C1IR,                                    \
    ( const arrtype* src, int srcstep, acctype* dst, int dststep, CvSize size ))    \
IPCVAPI( CvStatus, icvAddProduct_##flavor##_C1IR,                                   \
    ( const arrtype* src1, int srcstep1, const arrtype* src2, int srcstep2,         \
      acctype* dst, int dststep, CvSize size ))                                     \
IPCVAPI( CvStatus, icvAddWeighted_##flavor##_C1IR,                                  \
    ( const arrtype* src, int srcstep, acctype* dst, int dststep,                   \
      CvSize size, acctype alpha ))                                                 \
                                                                                    \
IPCVAPI( CvStatus, icvAdd_##flavor##_C1IMR,                                         \
    ( const arrtype* src, int srcstep, const uchar* mask, int maskstep,             \
      acctype* dst, int dststep, CvSize size ))                                     \
IPCVAPI( CvStatus, icvAddSquare_##flavor##_C1IMR,                                   \
    ( const arrtype* src, int srcstep, const uchar* mask, int maskstep,             \
      acctype* dst, int dststep, CvSize size ))                                     \
IPCVAPI( CvStatus, icvAddProduct_##flavor##_C1IMR,                                  \
    ( const arrtype* src1, int srcstep1, const arrtype* src2, int srcstep2,         \
      const uchar* mask, int maskstep, acctype* dst, int dststep, CvSize size ))    \
IPCVAPI( CvStatus, icvAddWeighted_##flavor##_C1IMR,                                 \
    ( const arrtype* src, int srcstep, const uchar* mask, int maskstep,             \
      acctype* dst, int dststep, CvSize size, acctype alpha ))                      \
                                                                                    \
IPCVAPI( CvStatus, icvAdd_##flavor##_C3IMR,                                         \
    ( const arrtype* src, int srcstep, const uchar* mask, int maskstep,             \
      acctype* dst, int dststep, CvSize size ))                                     \
IPCVAPI( CvStatus, icvAddSquare_##flavor##_C3IMR,                                   \
    ( const arrtype* src, int srcstep, const uchar* mask, int maskstep,             \
      acctype* dst, int dststep, CvSize size ))                                     \
IPCVAPI( CvStatus, icvAddProduct_##flavor##_C3IMR,                                  \
    ( const arrtype* src1, int srcstep1, const arrtype* src2, int srcstep2,         \
      const uchar* mask, int maskstep, acctype* dst, int dststep, CvSize size ))    \
IPCVAPI( CvStatus, icvAddWeighted_##flavor##_C3IMR,                                 \
    ( const arrtype* src, int srcstep, const uchar* mask, int maskstep,             \
      acctype* dst, int dststep, CvSize size, acctype alpha ))


IPCV_ACCUM( 8u32f, uchar, float )
IPCV_ACCUM( 8s32f, char, float )
IPCV_ACCUM( 32f, float, float )

#undef IPCV_ACCUM

/****************************************************************************************\
*                                         Samplers                                       *
\****************************************************************************************/

////////////////////////////////////// GetRectSubPix ////////////////////////////////////////

#define IPCV_GET_RECT_SUB_PIX( flavor, cn, srctype, dsttype )               \
IPCVAPI( CvStatus, icvGetRectSubPix_##flavor##_C##cn##R,                    \
( const srctype* src, int src_step, CvSize src_size,                        \
  dsttype* dst, int dst_step, CvSize win_size, CvPoint2D32f center ))

IPCV_GET_RECT_SUB_PIX( 8u, 1, uchar, uchar )
IPCV_GET_RECT_SUB_PIX( 8u32f, 1, uchar, float )
IPCV_GET_RECT_SUB_PIX( 32f, 1, float, float )

IPCV_GET_RECT_SUB_PIX( 8u, 3, uchar, uchar )
IPCV_GET_RECT_SUB_PIX( 8u32f, 3, uchar, float )
IPCV_GET_RECT_SUB_PIX( 32f, 3, float, float )

#define IPCV_GET_QUANDRANGLE_SUB_PIX( flavor, cn, srctype, dsttype )  \
IPCVAPI( CvStatus, icvGetQuadrangleSubPix_##flavor##_C##cn##R,    \
( const srctype* src, int src_step, CvSize src_size,              \
  dsttype* dst, int dst_step, CvSize win_size,                    \
  const float *matrix, int fillOutliers, dsttype* fillValue ))

IPCV_GET_QUANDRANGLE_SUB_PIX( 8u, 1, uchar, uchar )
IPCV_GET_QUANDRANGLE_SUB_PIX( 8u32f, 1, uchar, float )
IPCV_GET_QUANDRANGLE_SUB_PIX( 32f, 1, float, float )

IPCV_GET_QUANDRANGLE_SUB_PIX( 8u, 3, uchar, uchar )
IPCV_GET_QUANDRANGLE_SUB_PIX( 8u32f, 3, uchar, float )
IPCV_GET_QUANDRANGLE_SUB_PIX( 32f, 3, float, float )

#undef IPCV_GET_RECT_SUB_PIX
#undef IPCV_GET_QUANDRANGLE_SUB_PIX


/****************************************************************************************\
*                                        Pyramids                                        *
\****************************************************************************************/

IPCVAPI( CvStatus, icvPyrUpGetBufSize_Gauss5x5, ( int roiWidth, CvDataType dataType,
                                                  int channels, int* bufSize))

IPCVAPI( CvStatus, icvPyrDownGetBufSize_Gauss5x5, (int roiWidth, CvDataType dataType,
                                                   int channels, int* bufSize))

#define ICV_PYRAMID( name, flavor, arrtype )                    \
IPCVAPI( CvStatus, icv##name##_##flavor##_C1R,                  \
( const arrtype* pSrc, int srcStep, arrtype* pDst, int dstStep, \
  CvSize roiSize, void* pBuffer ))                              \
IPCVAPI( CvStatus, icv##name##_##flavor##_C3R,                  \
( const arrtype* pSrc, int srcStep, arrtype* pDst, int dstStep, \
  CvSize roiSize, void* pBuffer ))

ICV_PYRAMID( PyrUp_Gauss5x5, 8u, uchar )
ICV_PYRAMID( PyrUp_Gauss5x5, 8s, char )
ICV_PYRAMID( PyrUp_Gauss5x5, 32f, float )
ICV_PYRAMID( PyrUp_Gauss5x5, 64f, double )

ICV_PYRAMID( PyrDown_Gauss5x5, 8u, uchar )
ICV_PYRAMID( PyrDown_Gauss5x5, 8s, char )
ICV_PYRAMID( PyrDown_Gauss5x5, 32f, float )
ICV_PYRAMID( PyrDown_Gauss5x5, 64f, double )

#undef ICV_PYRAMID

/****************************************************************************************/
/*                              Morphological primitives                                */
/****************************************************************************************/

/****************************************************************************************/
/*                                  Erosion primitives                                  */
/****************************************************************************************/

IPCVAPI(CvStatus, icvErodeStrip_Rect_8u_C1R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Rect_8u_C3R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Rect_8u_C4R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Rect_32f_C1R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Rect_32f_C3R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Rect_32f_C4R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Cross_8u_C1R, ( const uchar* pSrc, int srcStep,
                                            uchar* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Cross_8u_C3R, ( const uchar* pSrc, int srcStep,
                                            uchar* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Cross_8u_C4R, ( const uchar* pSrc, int srcStep,
                                            uchar* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Cross_32f_C1R, ( const float* pSrc, int srcStep,
                                             float* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Cross_32f_C3R, ( const float* pSrc, int srcStep,
                                             float* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvErodeStrip_Cross_32f_C4R, ( const float* pSrc, int srcStep,
                                             float* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvErodeStrip_8u_C1R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvErodeStrip_8u_C3R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvErodeStrip_8u_C4R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvErodeStrip_32f_C1R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_32f_C3R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvErodeStrip_32f_C4R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

/* /////////// Dilation //////////// */
IPCVAPI(CvStatus, icvDilateStrip_Rect_8u_C1R, ( const uchar* pSrc, int srcStep,
                                            uchar* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Rect_8u_C3R, ( const uchar* pSrc, int srcStep,
                                            uchar* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Rect_8u_C4R, ( const uchar* pSrc, int srcStep,
                                            uchar* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Rect_32f_C1R, ( const float* pSrc, int srcStep,
                                             float* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Rect_32f_C3R, ( const float* pSrc, int srcStep,
                                             float* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Rect_32f_C4R, ( const float* pSrc, int srcStep,
                                             float* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Cross_8u_C1R, ( const uchar* pSrc, int srcStep,
                                             uchar* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Cross_8u_C3R, ( const uchar* pSrc, int srcStep,
                                             uchar* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Cross_8u_C4R, ( const uchar* pSrc, int srcStep,
                                             uchar* pDst, int dstStep,
                                             CvSize* roiSize,
                                             struct CvMorphState* state,
                                             int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Cross_32f_C1R, ( const float* pSrc, int srcStep,
                                              float* pDst, int dstStep,
                                              CvSize* roiSize,
                                              struct CvMorphState* state,
                                              int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Cross_32f_C3R, ( const float* pSrc, int srcStep,
                                              float* pDst, int dstStep,
                                              CvSize* roiSize,
                                              struct CvMorphState* state,
                                              int stage ))

IPCVAPI(CvStatus, icvDilateStrip_Cross_32f_C4R, ( const float* pSrc, int srcStep,
                                              float* pDst, int dstStep,
                                              CvSize* roiSize,
                                              struct CvMorphState* state,
                                              int stage ))

IPCVAPI(CvStatus, icvDilateStrip_8u_C1R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvDilateStrip_8u_C3R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvDilateStrip_8u_C4R, ( const uchar* pSrc, int srcStep,
                                           uchar* pDst, int dstStep,
                                           CvSize* roiSize,
                                           struct CvMorphState* state,
                                           int stage ))

IPCVAPI(CvStatus, icvDilateStrip_32f_C1R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvDilateStrip_32f_C3R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvDilateStrip_32f_C4R, ( const float* pSrc, int srcStep,
                                            float* pDst, int dstStep,
                                            CvSize* roiSize,
                                            struct CvMorphState* state,
                                            int stage ))

IPCVAPI(CvStatus, icvMorphologyInitAlloc, ( int roiWidth,
                                            CvDataType dataType, int channels,
                                            CvSize elSize, CvPoint elAnchor,
                                            CvElementShape elShape, int* elData,
                                            struct CvMorphState** morphState ))

IPCVAPI(CvStatus, icvMorphologyFree, ( struct CvMorphState** morphState ))

/****************************************************************************************/
/*                                  Motion Template                                     */
/****************************************************************************************/

IPCVAPI( CvStatus , icvUpdateMotionHistory_8u32f_C1IR, ( const uchar* silIm, int silStep,
                                                         float* mhiIm, int mhiStep,
                                                         CvSize size,
                                                         float  timestamp,
                                                         float  mhi_duration ))

/****************************************************************************************\
*                                      Template matching                                 *
\****************************************************************************************/

IPCVAPI( CvStatus, icvMatchTemplateGetBufSize_SqDiff, ( CvSize roiSize,
                                                        CvSize templSize,
                                                        CvDataType dataType,
                                                        int* bufferSize ))

IPCVAPI( CvStatus, icvMatchTemplateGetBufSize_SqDiffNormed, ( CvSize roiSize,
                                                              CvSize templSize,
                                                              CvDataType dataType,
                                                              int* bufferSize))

IPCVAPI( CvStatus, icvMatchTemplateGetBufSize_Corr, ( CvSize roiSize,
                                                      CvSize templSize,
                                                      CvDataType dataType,
                                                      int* bufferSize ))

IPCVAPI( CvStatus, icvMatchTemplateGetBufSize_CorrNormed, ( CvSize roiSize,
                                                            CvSize templSize,
                                                            CvDataType dataType,
                                                            int* bufferSize ))

IPCVAPI( CvStatus, icvMatchTemplateGetBufSize_Coeff, ( CvSize roiSize,
                                                       CvSize templSize,
                                                       CvDataType dataType,
                                                       int* bufferSize ))

IPCVAPI( CvStatus, icvMatchTemplateGetBufSize_CoeffNormed, ( CvSize roiSize,
                                                             CvSize templSize,
                                                             CvDataType dataType,
                                                             int* bufferSize ))


/*  processing functions */
IPCVAPI( CvStatus, icvMatchTemplate_SqDiff_8u32f_C1R,
                                (const uchar* pImage, int imageStep, CvSize roiSize,
                                 const uchar* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_SqDiffNormed_8u32f_C1R,
                                (const uchar* pImage, int imageStep, CvSize roiSize,
                                 const uchar* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_Corr_8u32f_C1R,
                                (const uchar* pImage, int imageStep, CvSize roiSize,
                                 const uchar* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_CorrNormed_8u32f_C1R,
                                (const uchar* pImage, int imageStep, CvSize roiSize,
                                 const uchar* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_Coeff_8u32f_C1R,
                                (const uchar* pImage, int imageStep, CvSize roiSize,
                                 const uchar* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_CoeffNormed_8u32f_C1R,
                                (const uchar* pImage, int imageStep, CvSize roiSize,
                                 const uchar* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_SqDiff_8s32f_C1R,
                                (const char* pImage, int imageStep, CvSize roiSize,
                                 const char* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_SqDiffNormed_8s32f_C1R,
                                (const char* pImage, int imageStep, CvSize roiSize,
                                 const char* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_Corr_8s32f_C1R,
                                (const char* pImage, int imageStep, CvSize roiSize,
                                 const char* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_CorrNormed_8s32f_C1R,
                                (const char* pImage, int imageStep, CvSize roiSize,
                                 const char* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_Coeff_8s32f_C1R,
                                (const char* pImage, int imageStep, CvSize roiSize,
                                 const char* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_CoeffNormed_8s32f_C1R,
                                (const char* pImage, int imageStep, CvSize roiSize,
                                 const char* pTemplate, int templStep, CvSize templSize,
                                 float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_SqDiff_32f_C1R,
                               (const float* pImage, int imageStep, CvSize roiSize,
                                const float* pTemplate, int templStep, CvSize templSize,
                                float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_SqDiffNormed_32f_C1R,
                               (const float* pImage, int imageStep, CvSize roiSize,
                                const float* pTemplate, int templStep, CvSize templSize,
                                float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_Corr_32f_C1R,
                               (const float* pImage, int imageStep, CvSize roiSize,
                                const float* pTemplate, int templStep, CvSize templSize,
                                float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_CorrNormed_32f_C1R,
                               (const float* pImage, int imageStep, CvSize roiSize,
                                const float* pTemplate, int templStep, CvSize templSize,
                                float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_Coeff_32f_C1R,
                               (const float* pImage, int imageStep, CvSize roiSize,
                                const float* pTemplate, int templStep, CvSize templSize,
                                float* pResult, int resultStep, void* pBuffer ))

IPCVAPI( CvStatus, icvMatchTemplate_CoeffNormed_32f_C1R,
                               (const float* pImage, int imageStep, CvSize roiSize,
                                const float* pTemplate, int templStep, CvSize templSize,
                                float* pResult, int resultStep, void* pBuffer ))


/****************************************************************************************/
/*                                Distance Transform                                    */
/****************************************************************************************/

IPCVAPI(CvStatus, icvDistanceTransform_3x3_8u32f_C1R, ( const uchar* pSrc, int srcStep,
                                                         float* pDst, int dstStep,
                                                         CvSize roiSize, float* pMetrics ))

IPCVAPI(CvStatus, icvDistanceTransform_5x5_8u32f_C1R, ( const uchar* pImage, int imgStep,
                                                        float* pDist, int distStep,
                                                        CvSize roiSize, float* pMetrics ))

IPCVAPI( CvStatus, icvGetDistanceTransformMask, ( int maskType, float* pMetrics ))


/****************************************************************************************/
/*                            Math routines and RNGs                                    */
/****************************************************************************************/

IPCVAPI( CvStatus, icvbInvSqrt_32f, ( const float* src, float* dst, int len ))
IPCVAPI( CvStatus, icvbSqrt_32f, ( const float* src, float* dst, int len ))
IPCVAPI( CvStatus, icvbInvSqrt_64f, ( const double* src, double* dst, int len ))
IPCVAPI( CvStatus, icvbSqrt_64f, ( const double* src, double* dst, int len ))

IPCVAPI( CvStatus, icvbLog_64f32f, ( const double *x, float *y, int n ) )
IPCVAPI( CvStatus, icvbExp_32f64f, ( const float *x, double *y, int n ) )
IPCVAPI( CvStatus, icvbFastArctan_32f, ( const float* y, const float* x,
                                         float* angle, int len ))

IPCVAPI(CvStatus, icvMinEigenValGetSize, ( int roiWidth,
                                           int apertureSize, int avgWindow,
                                           int* bufferSize ))

IPCVAPI(CvStatus, icvMinEigenVal_8u32f_C1R, ( const unsigned char* pSrc, int srcStep,
                                               float* pMinEigenVal, int minValStep,
                                               CvSize roiSize, int apertureSize,
                                               int avgWindow, void* pBuffer ))

IPCVAPI(CvStatus, icvMinEigenVal_8s32f_C1R, ( const char* pSrc, int srcStep,
                                               float* pMinEigenVal, int minValStep,
                                               CvSize roiSize, int apertureSize,
                                               int avgWindow, void* pBuffer ))

IPCVAPI(CvStatus, icvMinEigenVal_32f_C1R, ( const float* pSrc, int srcStep,
                                             float* pMinEigenVal, int minValStep,
                                             CvSize roiSize, int apertureSize,
                                             int avgWindow, void* pBuffer ))

/****************************************************************************************/
/*                                  Pyramid segmentation                                */
/****************************************************************************************/

IPCVAPI( CvStatus,  icvUpdatePyrLinks_8u_C1, (
                               int     layer,
                               void*   layer_data,
                               CvSize  size,
                               void*   parent_layer,
                               void*   writer,  
                               float   threshold,
                               int     is_last_iter,
                               void*   stub,
                               ICVWriteNodeFunction ))

IPCVAPI( CvStatus,  icvUpdatePyrLinks_8u_C3, (
                               int     layer,
                               void*   layer_data,
                               CvSize  size,
                               void*   parent_layer,
                               void*   writer,
                               float   threshold,
                               int     is_last_iter,
                               void*   stub,
                               ICVWriteNodeFunction ))

/****************************************************************************************\
*                                      Lens undistortion                                 *
\****************************************************************************************/

IPCVAPI( CvStatus, icvUnDistort1_8u_C1R, ( const uchar* srcImage, int srcStep,
                                           uchar* dstImage, int dstStep,
                                           CvSize size, const float* intrMatrix,
                                           const float* distCoeffs, int interToggle ))

IPCVAPI( CvStatus, icvUnDistort1_8u_C3R, ( const uchar* srcImage, int srcStep,
                                           uchar* dstImage, int dstStep,
                                           CvSize size, const float* intrMatrix,
                                           const float* distCoeffs, int interToggle ))

IPCVAPI( CvStatus, icvUnDistortEx_8u_C1R, ( const uchar* srcImage, int srcStep,
                                            uchar* dstImage, int dstStep,
                                            CvSize size, const float* intrMatrix,
                                            const float* distCoeffs, int interToggle ))

IPCVAPI( CvStatus, icvUnDistortEx_8u_C3R, ( const uchar* srcImage, int srcStep,
                                            uchar* dstImage, int dstStep,
                                            CvSize size, const float* intrMatrix,
                                            const float* distCoeffs, int interToggle ))

IPCVAPI( CvStatus, icvUnDistortInit, ( int srcStep, int* map,
                                       int mapStep, CvSize size,
                                       const float* intrMatrix,
                                       const float* distCoeffs,
                                       int interToggle, int pixSize ))

IPCVAPI( CvStatus, icvUnDistort_8u_C1R, ( const uchar* srcImage, int srcStep,
                                          const int* map, int mapstep,
                                          uchar* dstImage, int dstStep,
                                          CvSize size, int interToggle ))

IPCVAPI( CvStatus, icvUnDistort_8u_C3R, ( const uchar* srcImage, int srcStep,
                                          const int* map, int mapstep,
                                          uchar* dstImage, int dstStep,
                                          CvSize size, int interToggle ))

/****************************************************************************************\
*                                  Error handling functions                              *
\****************************************************************************************/

IPCVAPI( CVStatus, icvErrorFromStatus,( CvStatus status ) )


IPCVAPI( CvStatus, icvCheckArray_32f_C1R, ( const float* src, int srcstep,
                                 CvSize size, int flags,
                                 double min_val, double max_val ))

IPCVAPI( CvStatus, icvCheckArray_64f_C1R, ( const double* src, int srcstep,
                                 CvSize size, int flags,
                                 double min_val, double max_val ))

/****************************************************************************************/
/*                             HMM (Hidden Markov Models)                               */
/****************************************************************************************/

IPCVAPI( float, icvSquareDistance, ( CvVect32f v1, CvVect32f v2, int len ) )
                                   
IPCVAPI( CvStatus, icvViterbiSegmentation, ( int num_states, int num_obs,
                                    CvMatr32f transP,
                                    CvMatr32f B, /*muDur[0], */
                                    int start_obs, int prob_type,
                                    int** q, int min_num_obs, int max_num_obs,
                                    float* prob  ) )

IPCVAPI( CvStatus, icvInvertMatrix_32f, ( const float* src, int w, float* dst ))
IPCVAPI( CvStatus, icvInvertMatrix_64d, ( const double* src, int w, double* dst ))

#endif /*__IPCV_H_*/

