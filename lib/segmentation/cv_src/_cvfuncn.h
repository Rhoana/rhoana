/* ////////////////////////////////////////////////////////////////////
//
//  CvMat internal header for 0-ary (1 array), unary (2 arrays),
//  binary (3 arrays) and ternary (4 arrays) operations
//
// */

#ifndef __CVFUNCN_H__
#define __CVFUNCN_H__

typedef struct CvFuncTable
{
    void*   fn_2d[CV_DEPTH_MAX + 1];
}
CvFuncTable;


typedef struct CvBigFuncTable
{
    void*   fn_2d[CV_MAT_TYPE_MASK+1];
}
CvBigFuncTable;


typedef struct CvBtFuncTable
{
    void*   fn_2d[33];
}
CvBtFuncTable;


typedef  CvStatus (CV_STDCALL *CvArithmBinMaskFunc2D)( const void* src1, int step1,
                                            void* dst, int step,
                                            const void* mask, int maskstep,
                                            CvSize size, int cn );

typedef  CvStatus (CV_STDCALL *CvArithmUniMaskFunc2D)( void* dst, int step,
                                            const void* mask, int maskstep,
                                            CvSize, const void* val, int cn );

typedef CvStatus (CV_STDCALL *CvFunc1D_1A1P)( void* arr, int len, void* param );

typedef CvStatus (CV_STDCALL *CvFunc1D_2A1P)( void* arr1, void* arr2, int len, void* param);

typedef CvStatus (CV_STDCALL *CvFunc1D_3A)( void* arr1, void* arr2, void* arr3, int len );

typedef CvStatus (CV_STDCALL *CvFunc2D_1A)(void* arr, int step, CvSize size);

typedef CvStatus (CV_STDCALL *CvFunc2D_1A1P)(void* arr, int step, CvSize size, void* param);

typedef CvStatus (CV_STDCALL *CvFunc2DnC_1A1P)( void* arr, int step, CvSize size,
                                                int cn, int coi, void* param );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_1A1P)( void* arr, int step, CvSize size,
                                                int cn, int coi, void* param );

typedef CvStatus (CV_STDCALL *CvFunc2D_1A2P)( void* arr, int step, CvSize size,
                                              void* param1, void* param2 );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_1A2P)( void* arr, int step,
                                                CvSize size, int cn, int coi,
                                                void* param1, void* param2 );

typedef CvStatus (CV_STDCALL *CvFunc2D_1A4P)( void* arr, int step, CvSize size,
                                              void* param1, void* param2,
                                              void* param3, void* param4 );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_1A4P)( void* arr, int step,
                                                CvSize size, int cn, int coi,
                                                void* param1, void* param2,
                                                void* param3, void* param4 );

typedef CvStatus (CV_STDCALL *CvFunc2D_2A)( void* arr0, int step0,
                                            void* arr1, int step1, CvSize size );

typedef CvStatus (CV_STDCALL *CvFunc2D_2A1P)( void* arr0, int step0,
                                              void* arr1, int step1,
                                              CvSize size, void* param );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_2A1P)( void* arr0, int step0,
                                                void* arr1, int step1,
                                                CvSize size, int cn,
                                                int coi, void* param );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_2A1P)( void* arr0, int step0,
                                                void* arr1, int step1,
                                                CvSize size, int cn,
                                                int coi, void* param );

typedef CvStatus (CV_STDCALL *CvFunc2D_2A2P)( void* arr0, int step0,
                                              void* arr1, int step1, CvSize size,
                                              void* param1, void* param2 );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_2A2P)( void* arr0, int step0,
                                                void* arr1, int step1,
                                                CvSize size, int cn, int coi,
                                                void* param1, void* param2 );

typedef CvStatus (CV_STDCALL *CvFunc2D_2A1P1I)( void* arr0, int step0,
                                                void* arr1, int step1, CvSize size,
                                                void* param, int flag );

typedef CvStatus (CV_STDCALL *CvFunc2D_2A4P)( void* arr0, int step0,
                                              void* arr1, int step1, CvSize size,
                                              void* param1, void* param2,
                                              void* param3, void* param4 );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_2A4P)( void* arr0, int step0,
                                                void* arr1, int step1, CvSize size,
                                                int cn, int coi,
                                                void* param1, void* param2,
                                                void* param3, void* param4 );

typedef CvStatus (CV_STDCALL *CvFunc2D_3A)( void* arr0, int step0,
                                            void* arr1, int step1,
                                            void* arr2, int step2, CvSize size );

typedef CvStatus (CV_STDCALL *CvFunc2D_3A1P)( void* arr0, int step0,
                                              void* arr1, int step1,
                                              void* arr2, int step2,
                                              CvSize size, void* param );

typedef CvStatus (CV_STDCALL *CvFunc2DnC_3A1P)( void* arr0, int step0,
                                                void* arr1, int step1,
                                                void* arr2, int step2,
                                                CvSize size, int cn,
                                                int coi, void* param );

typedef CvStatus (CV_STDCALL *CvFunc2D_4A)( void* arr0, int step0,
                                            void* arr1, int step1,
                                            void* arr2, int step2,
                                            void* arr3, int step3,
                                            CvSize size );

typedef CvStatus (CV_STDCALL *CvFunc0D)( const void* src, void* dst, int param );


#define CV_DEF_INIT_FUNC_TAB( FUNCNAME, FLAG )                      \
static void  icvInit##FUNCNAME##FLAG##Table( CvFuncTable* tab )     \
{                                                                   \
    assert( tab );                                                  \
                                                                    \
    tab->fn_2d[CV_8U]  = (void*)icv##FUNCNAME##_8u_##FLAG;          \
    tab->fn_2d[CV_8S]  = (void*)icv##FUNCNAME##_8s_##FLAG;          \
    tab->fn_2d[CV_16S] = (void*)icv##FUNCNAME##_16s_##FLAG;         \
    tab->fn_2d[CV_32S] = (void*)icv##FUNCNAME##_32s_##FLAG;         \
    tab->fn_2d[CV_32F] = (void*)icv##FUNCNAME##_32f_##FLAG;         \
    tab->fn_2d[CV_64F] = (void*)icv##FUNCNAME##_64f_##FLAG;         \
}


#define CV_DEF_INIT_FUNC_TAB_2D( FUNCNAME, FLAG )                   \
static void  icvInit##FUNCNAME##FLAG##Table( CvFuncTable* tab )     \
{                                                                   \
    assert( tab );                                                  \
                                                                    \
    tab->fn_2d[CV_8U]  = (void*)icv##FUNCNAME##_8u_##FLAG;          \
    tab->fn_2d[CV_8S]  = (void*)icv##FUNCNAME##_8s_##FLAG;          \
    tab->fn_2d[CV_16S] = (void*)icv##FUNCNAME##_16s_##FLAG;         \
    tab->fn_2d[CV_32S] = (void*)icv##FUNCNAME##_32s_##FLAG;         \
    tab->fn_2d[CV_32F] = (void*)icv##FUNCNAME##_32f_##FLAG;         \
    tab->fn_2d[CV_64F] = (void*)icv##FUNCNAME##_64f_##FLAG;         \
}


#define CV_DEF_INIT_BIG_FUNC_TAB( FUNCNAME, FLAG )                  \
static void  icvInit##FUNCNAME##FLAG##Table( CvBigFuncTable* tab )  \
{                                                                   \
    assert( tab );                                                  \
                                                                    \
    tab->fn_2d[CV_8UC1]  = (void*)icv##FUNCNAME##_8u_C1##FLAG;      \
    tab->fn_2d[CV_8UC2]  = (void*)icv##FUNCNAME##_8u_C2##FLAG;      \
    tab->fn_2d[CV_8UC3]  = (void*)icv##FUNCNAME##_8u_C3##FLAG;      \
    tab->fn_2d[CV_8UC4]  = (void*)icv##FUNCNAME##_8u_C4##FLAG;      \
                                                                    \
    tab->fn_2d[CV_8SC1]  = (void*)icv##FUNCNAME##_8s_C1##FLAG;      \
    tab->fn_2d[CV_8SC2]  = (void*)icv##FUNCNAME##_8s_C2##FLAG;      \
    tab->fn_2d[CV_8SC3]  = (void*)icv##FUNCNAME##_8s_C3##FLAG;      \
    tab->fn_2d[CV_8SC4]  = (void*)icv##FUNCNAME##_8s_C4##FLAG;      \
                                                                    \
    tab->fn_2d[CV_16SC1] = (void*)icv##FUNCNAME##_16s_C1##FLAG;     \
    tab->fn_2d[CV_16SC2] = (void*)icv##FUNCNAME##_16s_C2##FLAG;     \
    tab->fn_2d[CV_16SC3] = (void*)icv##FUNCNAME##_16s_C3##FLAG;     \
    tab->fn_2d[CV_16SC4] = (void*)icv##FUNCNAME##_16s_C4##FLAG;     \
                                                                    \
    tab->fn_2d[CV_32SC1] = (void*)icv##FUNCNAME##_32s_C1##FLAG;     \
    tab->fn_2d[CV_32SC2] = (void*)icv##FUNCNAME##_32s_C2##FLAG;     \
    tab->fn_2d[CV_32SC3] = (void*)icv##FUNCNAME##_32s_C3##FLAG;     \
    tab->fn_2d[CV_32SC4] = (void*)icv##FUNCNAME##_32s_C4##FLAG;     \
                                                                    \
    tab->fn_2d[CV_32FC1] = (void*)icv##FUNCNAME##_32f_C1##FLAG;     \
    tab->fn_2d[CV_32FC2] = (void*)icv##FUNCNAME##_32f_C2##FLAG;     \
    tab->fn_2d[CV_32FC3] = (void*)icv##FUNCNAME##_32f_C3##FLAG;     \
    tab->fn_2d[CV_32FC4] = (void*)icv##FUNCNAME##_32f_C4##FLAG;     \
                                                                    \
    tab->fn_2d[CV_64FC1] = (void*)icv##FUNCNAME##_64f_C1##FLAG;     \
    tab->fn_2d[CV_64FC2] = (void*)icv##FUNCNAME##_64f_C2##FLAG;     \
    tab->fn_2d[CV_64FC3] = (void*)icv##FUNCNAME##_64f_C3##FLAG;     \
    tab->fn_2d[CV_64FC4] = (void*)icv##FUNCNAME##_64f_C4##FLAG;     \
}

#define CV_DEF_INIT_BIG_FUNC_TAB_2D CV_DEF_INIT_BIG_FUNC_TAB


#define CV_DEF_INIT_BIG_FUNC_TAB_C1_AND_COI( FUNCNAME, FLAG)        \
static void  icvInit##FUNCNAME##FLAG##CoiTable( CvBigFuncTable* tab)\
{                                                                   \
    assert( tab );                                                  \
                                                                    \
    tab->fn_2d[CV_8UC1]  = (void*)icv##FUNCNAME##_8u_C1##FLAG;      \
    tab->fn_2d[CV_8UC2]  = (void*)icv##FUNCNAME##_8u_C2C##FLAG;     \
    tab->fn_2d[CV_8UC3]  = (void*)icv##FUNCNAME##_8u_C3C##FLAG;     \
    tab->fn_2d[CV_8UC4]  = (void*)icv##FUNCNAME##_8u_C4C##FLAG;     \
                                                                    \
    tab->fn_2d[CV_8SC1]  = (void*)icv##FUNCNAME##_8s_C1##FLAG;      \
    tab->fn_2d[CV_8SC2]  = (void*)icv##FUNCNAME##_8s_C2C##FLAG;     \
    tab->fn_2d[CV_8SC3]  = (void*)icv##FUNCNAME##_8s_C3C##FLAG;     \
    tab->fn_2d[CV_8SC4]  = (void*)icv##FUNCNAME##_8s_C4C##FLAG;     \
                                                                    \
    tab->fn_2d[CV_16SC1] = (void*)icv##FUNCNAME##_16s_C1##FLAG;     \
    tab->fn_2d[CV_16SC2] = (void*)icv##FUNCNAME##_16s_C2C##FLAG;    \
    tab->fn_2d[CV_16SC3] = (void*)icv##FUNCNAME##_16s_C3C##FLAG;    \
    tab->fn_2d[CV_16SC4] = (void*)icv##FUNCNAME##_16s_C4C##FLAG;    \
                                                                    \
    tab->fn_2d[CV_32SC1] = (void*)icv##FUNCNAME##_32s_C1##FLAG;     \
    tab->fn_2d[CV_32SC2] = (void*)icv##FUNCNAME##_32s_C2C##FLAG;    \
    tab->fn_2d[CV_32SC3] = (void*)icv##FUNCNAME##_32s_C3C##FLAG;    \
    tab->fn_2d[CV_32SC4] = (void*)icv##FUNCNAME##_32s_C4C##FLAG;    \
                                                                    \
    tab->fn_2d[CV_32FC1] = (void*)icv##FUNCNAME##_32f_C1##FLAG;     \
    tab->fn_2d[CV_32FC2] = (void*)icv##FUNCNAME##_32f_C2C##FLAG;    \
    tab->fn_2d[CV_32FC3] = (void*)icv##FUNCNAME##_32f_C3C##FLAG;    \
    tab->fn_2d[CV_32FC4] = (void*)icv##FUNCNAME##_32f_C4C##FLAG;    \
                                                                    \
    tab->fn_2d[CV_64FC1] = (void*)icv##FUNCNAME##_64f_C1##FLAG;     \
    tab->fn_2d[CV_64FC2] = (void*)icv##FUNCNAME##_64f_C2C##FLAG;    \
    tab->fn_2d[CV_64FC3] = (void*)icv##FUNCNAME##_64f_C3C##FLAG;    \
    tab->fn_2d[CV_64FC4] = (void*)icv##FUNCNAME##_64f_C4C##FLAG;    \
}


#define CV_DEF_INIT_FUNC_TAB_0D( FUNCNAME )                         \
static void  icvInit##FUNCNAME##Table( CvFuncTable* tab )           \
{                                                                   \
    tab->fn_2d[CV_8U]  = (void*)icv##FUNCNAME##_8u;                 \
    tab->fn_2d[CV_8S]  = (void*)icv##FUNCNAME##_8s;                 \
    tab->fn_2d[CV_16S] = (void*)icv##FUNCNAME##_16s;                \
    tab->fn_2d[CV_32S] = (void*)icv##FUNCNAME##_32s;                \
    tab->fn_2d[CV_32F] = (void*)icv##FUNCNAME##_32f;                \
    tab->fn_2d[CV_64F] = (void*)icv##FUNCNAME##_64f;                \
}

#define CV_DEF_INIT_FUNC_TAB_1D  CV_DEF_INIT_FUNC_TAB_0D


#define CV_DEF_INIT_PIXSIZE_TAB_2D( FUNCNAME, FLAG )                \
static void icvInit##FUNCNAME##FLAG##Table( CvBtFuncTable* table )  \
{                                                                   \
    table->fn_2d[1]  = (void*)icv##FUNCNAME##_8u_C1##FLAG;                 \
    table->fn_2d[2]  = (void*)icv##FUNCNAME##_8u_C2##FLAG;                 \
    table->fn_2d[3]  = (void*)icv##FUNCNAME##_8u_C3##FLAG;                 \
    table->fn_2d[4]  = (void*)icv##FUNCNAME##_16u_C2##FLAG;                \
    table->fn_2d[6]  = (void*)icv##FUNCNAME##_16u_C3##FLAG;                \
    table->fn_2d[8]  = (void*)icv##FUNCNAME##_32s_C2##FLAG;                \
    table->fn_2d[12] = (void*)icv##FUNCNAME##_32s_C3##FLAG;                \
    table->fn_2d[16] = (void*)icv##FUNCNAME##_64s_C2##FLAG;                \
    table->fn_2d[24] = (void*)icv##FUNCNAME##_64s_C3##FLAG;                \
    table->fn_2d[32] = (void*)icv##FUNCNAME##_64s_C4##FLAG;                \
}


#define  CV_GET_FUNC_PTR( func, table_entry )  \
    func = (table_entry);                      \
                                               \
    if( !func )                                \
        CV_ERROR( CV_StsUnsupportedFormat, "" )

#endif/*__CVFUNCN_H__*/


