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

/* ////////////////////////////////////////////////////////////////////
//
//  CvMat internal interface file
//
// */

#ifndef __CVARR_H__
#define __CVARR_H__

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void cvCheckMatHeader( const CvMat* arr, const char* arrname,
                         const char* file, int line );

void  cvScalarToMat( CvScalar* scalar, int flags, CvMat* arr, void* data );

#define CV_CHECK_ARR( arr ) CV_CALL( cvCheckMatHeader( (arr), #arr, __FILE__, __LINE__ ));

#define CV_DEPTH_MAX  5
#define CV_CN_MAX     4 

extern const char icvDepthToType[];
extern const int icvTypeToDepth[];

#define icvIplToCvDepth( depth ) \
    icvDepthToType[(((depth) & 255) >> 2) + ((depth) < 0)]

#define icvCvToIplDepth( type )  \
    icvTypeToDepth[(type)]


/* general-purpose saturation macros */ 
#define CV_CAST_8U(t)    (uchar)( !((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)
#define CV_CAST_8S(t)    (char)( !(((t)+128) & ~255) ? (t) : (t) > 0 ? 127 : -128 )
#define CV_CAST_16S(t)   (short)( !(((t)+32768) & ~65535) ? (t) : (t) > 0 ? 32767 : -32768 )
#define CV_CAST_32S(t)   (int)(t)
#define CV_CAST_64S(t)   (int64)(t)
#define CV_CAST_32F(t)   (float)(t)
#define CV_CAST_64F(t)   (double)(t)

/* helper tables */
extern const int icvPixSize[];
extern const float icv8to32f[];

extern const uchar icvSaturate8u[];
extern const char  icvSaturate8s[];

#define CV_FAST_CAST_8U(t)   (assert(-256 <= (t) || (t) <= 512), icvSaturate8u[t+256])
#define CV_FAST_CAST_8S(t)   (assert(-256 <= (t) || (t) <= 256), icvSaturate8s[t+256])

#define CV_PASTE2(a,b) a##b
#define CV_PASTE(a,b) CV_PASTE2(a,b)

CV_INLINE  CvSize  icvGetMatSize( const CvMat* mat );
CV_INLINE  CvSize  icvGetMatSize( const CvMat* mat )
{
    CvSize size = { mat->width, mat->height };
    return size;
}

#include "_cvfuncn.h"

CV_INLINE  CvDataType icvDepthToDataType( int type );
CV_INLINE  CvDataType icvDepthToDataType( int type )
{
    return (CvDataType)(
            ((((int)cv8u)|((int)cv8s << 4)|((int)cv16s << 8)|
              ((int)cv32s << 12)|((int)cv32f << 16)|
              ((int)cv64f << 20)) >> CV_MAT_DEPTH(type)*4) & 15);
}


#define CV_MAX_ARR 10

typedef struct CvMatNDIterator
{
    int count; // number of arrays
    int dims; // number of dimensions to iterate
    CvSize size; // maximal common linear size: { width = size, height = 1 }
    int stack[CV_MAX_DIM];
    uchar* ptr[CV_MAX_ARR];
    CvMatND* hdr[CV_MAX_ARR];
}
CvMatNDIterator;

#define CV_NO_DEPTH_CHECK     1
#define CV_NO_CN_CHECK        2
#define CV_NO_SIZE_CHECK      4

// returns number of dimensions to iterate.
int icvPrepareArrayOp( int count, CvArr** arrs,
                       const CvArr* mask, CvMatND* stubs,
                       CvMatNDIterator* iterator,
                       int flags CV_DEFAULT(0) );

// returns zero value if iteration is finished, non-zero otherwise
int icvNextMatNDSlice( CvMatNDIterator* iterator );


//////////////////////////////// sparse arrays ////////////////////////////////////

// returns pointer to sparse array node
// (optionally can create the node if it doesn't exist)
uchar* icvGetNodePtr( CvSparseMat* mat, int* idx,
                      int* _type CV_DEFAULT(0),
                      int create_node CV_DEFAULT(0),
                      unsigned* precalc_hashval CV_DEFAULT(0));

// deletes node from sparse matrix
void icvDeleteNode( CvSparseMat* mat, int* idx,
                    unsigned* precalc_hashval CV_DEFAULT(0));

#endif/*__CVARR_H__*/
