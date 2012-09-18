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

#ifndef _CVDATASTRUCTS_H_
#define _CVDATASTRUCTS_H_

/**************** helper macros and functions for sequence/contour processing ***********/

#define _CV_GET_LAST_ELEM( seq, block ) \
    ((block)->data + ((block)->count - 1)*((seq)->elem_size))

extern const CvPoint icvCodeDeltas[];

/* faster method for positioning sequence reader */
//void  icvSetSequenceReaderPos( int index, CvSeqReader* reader );

/* curvature: 0 - 1-curvature, 1 - k-cosine curvature. */
CvStatus  icvApproximateChainTC89( CvChain*      chain,
                                   int header_size,
                                   CvMemStorage* storage,
                                   CvSeq**   contour,
                                   CvChainApproxMethod method );

int icvSliceLength( CvSlice slice, CvSeq* seq );

CvSeq* icvPointSeqFromMat( int seq_kind, const CvArr* mat,
                           CvContour* contour_header,
                           CvSeqBlock* block );

#define CV_ADJUST_EDGE_COUNT( count, seq )  \
    ((count) -= ((count) == (seq)->total && !CV_IS_SEQ_CLOSED(seq)))

#define CV_SWAP_ELEMS(a,b)            \
{                                     \
    int k;                            \
    for( k = 0; k < elem_size; k++ )  \
    {                                 \
        char t0 = (a)[k];             \
        char t1 = (b)[k];             \
        (a)[k] = t1;                  \
        (b)[k] = t0;                  \
    }                                 \
}

#define CV_IMPLEMENT2_SEQ_QSORT( func_name, T, less_than, user_data_type )       \
void func_name( CvSeq* seq, user_data_type aux )                                 \
{                                                                                \
    const int bubble_level = 8;                                                  \
    const int elem_size = sizeof(T);                                             \
                                                                                 \
    struct                                                                       \
    {                                                                            \
        int lb, ub;                                                              \
    }                                                                            \
    stack[48];                                                                   \
    int sp = 0;                                                                  \
    int length = seq->total;                                                     \
                                                                                 \
    CvSeqReader r_i, r_j;                                                        \
    T t;                                                                         \
                                                                                 \
    cvStartReadSeq( seq, &r_i );                                                 \
    cvStartReadSeq( seq, &r_j );                                                 \
                                                                                 \
    stack[0].lb = 0;                                                             \
    stack[0].ub = length - 1;                                                    \
                                                                                 \
    aux = aux;                                                                   \
                                                                                 \
    while( sp >= 0 )                                                             \
    {                                                                            \
        int lb = stack[sp].lb;                                                   \
        int ub = stack[sp--].ub;                                                 \
                                                                                 \
        for(;;)                                                                  \
        {                                                                        \
            int diff = ub - lb;                                                  \
            if( diff < bubble_level )                                            \
            {                                                                    \
                int i, j;                                                        \
                cvSetSeqReaderPos( &r_i, lb );                                   \
                                                                                 \
                for( i = diff; i > 0; i-- )                                      \
                {                                                                \
                    int f = 0;                                                   \
                    r_j.ptr = r_i.ptr;                                           \
                    r_j.block_min = r_i.block_min;                               \
                    r_j.block_max = r_i.block_max;                               \
                    r_j.block = r_i.block;                                       \
                                                                                 \
                    T* curr = (T*)(r_j.ptr);                                     \
                                                                                 \
                    for( j = 0; j < i; j++ )                                     \
                    {                                                            \
                        CV_NEXT_SEQ_ELEM( elem_size, r_j );                      \
                        T* next = (T*)(r_j.ptr);                                 \
                                                                                 \
                        if( less_than( *next, *curr ))                           \
                        {                                                        \
                            CV_SWAP( *curr, *next, t );                          \
                            f = 1;                                               \
                        }                                                        \
                        curr = next;                                             \
                    }                                                            \
                    if( !f ) break;                                              \
                }                                                                \
                break;                                                           \
            }                                                                    \
            else                                                                 \
            {                                                                    \
                /* select pivot and exchange with 1st element */                 \
                int  m = lb + (diff >> 1);                                       \
                int  i = lb + 1, j = ub;                                         \
                                                                                 \
                cvSetSeqReaderPos( &r_i, lb );                                   \
                T* lb_ptr = (T*)r_i.ptr;                                         \
                                                                                 \
                cvSetSeqReaderPos( &r_j, m );                                    \
                                                                                 \
                T lb_val = *(T*)(r_j.ptr);                                       \
                *(T*)(r_j.ptr) = *(T*)(r_i.ptr);                                 \
                                                                                 \
                CV_NEXT_SEQ_ELEM( elem_size, r_i );                              \
                cvSetSeqReaderPos( &r_j, ub );                                   \
                                                                                 \
                /* partition into two segments */                                \
                for(;;)                                                          \
                {                                                                \
                    for( ; i < j && less_than( *(T*)(r_i.ptr), lb_val ); i++ )   \
                    {                                                            \
                        CV_NEXT_SEQ_ELEM( elem_size, r_i );                      \
                    }                                                            \
                                                                                 \
                    for( ; j >= i && less_than( lb_val, *(T*)(r_j.ptr) ); j-- )  \
                    {                                                            \
                        CV_PREV_SEQ_ELEM( elem_size, r_j );                      \
                    }                                                            \
                                                                                 \
                    if( i >= j ) break;                                          \
                    CV_SWAP( *(T*)(r_i.ptr), *(T*)(r_j.ptr), t );                \
                    CV_NEXT_SEQ_ELEM( elem_size, r_i );                          \
                    CV_PREV_SEQ_ELEM( elem_size, r_j );                          \
                    i++, j--;                                                    \
                }                                                                \
                                                                                 \
                /* pivot belongs in A[j] */                                      \
                *lb_ptr = *(T*)(r_j.ptr);                                        \
                *(T*)(r_j.ptr) = lb_val;                                         \
                                                                                 \
                /* keep processing smallest segment, and stack largest*/         \
                if( j - lb <= ub - j )                                           \
                {                                                                \
                    if( j + 1 < ub )                                             \
                    {                                                            \
                        stack[++sp].lb   = j + 1;                                \
                        stack[sp].ub = ub;                                       \
                    }                                                            \
                    ub = j - 1;                                                  \
                }                                                                \
                else                                                             \
                {                                                                \
                    if( j - 1 > lb)                                              \
                    {                                                            \
                        stack[++sp].lb = lb;                                     \
                        stack[sp].ub = j - 1;                                    \
                    }                                                            \
                    lb = j + 1;                                                  \
                }                                                                \
            }                                                                    \
        }                                                                        \
    }                                                                            \
}


#define CV_IMPLEMENT_SEQ_QSORT( func_name, T, less_than )  \
    CV_IMPLEMENT2_SEQ_QSORT( func_name, T, less_than, int )

/*
  Single-connected list, based on CvSet.
*/
#define CV_LIST_ELEM_FIELDS() \
    struct CvSListNode* next;

typedef struct CvSListNode
{
    CV_LIST_ELEM_FIELDS()
}
CvSListNode;

#define CV_LIST_FIELDS() \
    CV_SEQUENCE_FIELDS() \
    int  nodes;          \
    CvSListNode* head;

typedef struct CvSList
{
    CV_LIST_FIELDS()
}
CvSList;

CvSList* icvCreateSList( int list_flags, int header_size, int elem_size, CvMemStorage* storage );
CvSListNode* icvSListGetNode( CvSList* list, int index, CvSListNode** prevNode CV_DEFAULT(0));
int  icvSListGetIndex( CvSList* list, CvSListNode* node, CvSListNode** prevNode CV_DEFAULT(0) );
void icvSListRemoveAfter( CvSList* list, CvSListNode* node );
CvSListNode* icvSListInsertAfter( CvSList* list, CvSListNode* node,
                                CvSListNode* newNode CV_DEFAULT(0));
void icvClearSList( CvSList* list );

#define icvSListAddHead( list, newNode )  icvSListInsertAfter( (list), 0, (newNode))

/*
  Double-connected list, based on CvSList.
*/
#define CV_DBLIST_ELEM_FIELDS() \
    struct CvListNode* next;  \
    struct CvListNode* prev;

typedef struct CvListNode
{
    CV_DBLIST_ELEM_FIELDS()
}
CvListNode;

#define CV_DBLIST_FIELDS() \
    CV_SEQUENCE_FIELDS()   \
    int  nodes;            \
    CvListNode* head;    \
    CvListNode* tail;

typedef struct CvList
{
    CV_DBLIST_FIELDS()
}
CvList;

CvList* icvCreateList( int list_flags, int header_size, int elem_size, CvMemStorage* storage );
CvListNode* icvListGetNode( CvSList* list, int index );
int  icvListGetIndex( CvList* list, CvListNode* node );
void icvListRemoveAfter( CvList* list, CvListNode* node );
void icvListRemoveBefore( CvList* list, CvListNode* node );
CvListNode* icvListInsertAfter( CvList* list, CvListNode* node,
                                    CvListNode* newNode CV_DEFAULT(0) );
CvListNode* icvListInsertBefore( CvList* list, CvListNode* node,
                                     CvListNode* newNode CV_DEFAULT(0) );
void icvClearList( CvList* list );

#define  icvListAddHead( list, newNode )  icvSListInsertAfter( (list), 0, (newNode))
#define  icvListAddTail( list, newNode )  icvSListInsertBefore( (list), 0, (newNode))

#endif/*_CVDATASTRUCTS_H_*/

/* End of file. */
