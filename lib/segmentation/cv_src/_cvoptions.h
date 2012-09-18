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

#ifndef _CVOPTIONS_H_
#define _CVOPTIONS_H_

// maximal size of vector to run matrix operations on it inline (i.e. w/o ipp calls)
#define  CV_MAX_INLINE_MAT_OP_SIZE  10

// maximal linear size of matrix to allocate it the stack.
#define  CV_MAX_LOCAL_MAT_SIZE  32

// maximal size of local memory storage
#define  CV_MAX_LOCAL_SIZE  \
    (CV_MAX_LOCAL_MAT_SIZE*CV_MAX_LOCAL_MAT_SIZE*(int)sizeof(double))

// default image row align
#define  CV_DEFAULT_ROW_ALIGN  4

// maximum size of dynamic memory buffer
#define  CV_MAX_ALLOC_SIZE    (((size_t)1 << (sizeof(size_t)*8-2)))

// boundary all the allocated buffers are aligned by
#define  CV_MALLOC_ALIGN  32

// default alignment for dynamic data strucutures, resided in storages.
#define  CV_STRUCT_ALIGN  ((int)sizeof(double))

// default step, set in case of continuous data
// to work around ipp functions checks for step
#define  CV_STUB_STEP     (1 << 30)

// default storage block size
#define  CV_STORAGE_BLOCK_SIZE   ((1<<16) - 128)

// default memory block for sparse array elements
#define  CV_SPARSE_MAT_BLOCK     (1<<12)

// initial hash table size
#define  CV_SPARSE_HASH_SIZE0    (1<<10)

// maximal average node_count/hash_size ratio beyond which hash table is resized
#define  CV_SPARSE_HASH_RATIO    4

// default type of histogram bins
#define  CV_HIST_DEFAULT_TYPE    CV_32F

// max length of strings
#define  CV_MAX_STRLEN  1024

// if defined, many array processing functions checks results for NaN's and Inf's.
/*#if defined _DEBUG || defined DEBUG
    #define  CV_CHECK_FOR_NANS
#endif*/

#endif/*_CVOPTIONS_H_*/
