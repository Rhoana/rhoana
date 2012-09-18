/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//            Intel License Agreement
//        For Open Source Computer Vision Library
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

#include "_cv.h"
#include <stdio.h>

#ifdef WIN32  
#include <windows.h>  
#else
#include <pthread.h>
#endif 

CvContext* icvCreateContext()
{
    CvContext* context = (CvContext*)malloc( sizeof( CvContext ) );

    context->CVErrMode    = CV_ErrModeLeaf;
    context->CVLastStatus = CV_StsOk;

#ifdef WIN32
   context->CVErrorFunc   = cvGuiBoxReport;
#else
   context->CVErrorFunc   = cvStdErrReport;
#endif

   /* below is stuff for profiling */
   context->CVStackCapacity = 100; /* let it be so*/
   context->CVStackSize = 0;
   context->CVStack =
       (CvStackRecord*)malloc( sizeof(CvStackRecord) * context->CVStackCapacity );
   return context;
}

void icvDestroyContext(CvContext* context)
{
    free(context->CVStack);
    free(context);
}

#ifdef WIN32
    DWORD g_TlsIndex = TLS_OUT_OF_INDEXES;
#else
    pthread_key_t g_TlsIndex;
#endif

CvContext* icvGetContext()
{
#ifdef CV_DLL
#ifdef WIN32
    CvContext* pContext = (CvContext*)TlsGetValue( g_TlsIndex );
    if( !pContext )
    {
    pContext = icvCreateContext();

    if( !pContext )
    {
        FatalAppExit( 0, "OpenCV. Problem to allocate memory for TLS OpenCV context." );
    }
    TlsSetValue( g_TlsIndex, pContext );
    }
    return pContext;
#else
    CvContext* pContext = (CvContext*)pthread_getspecific( g_TlsIndex );
    if( !pContext )
    {
    pContext = icvCreateContext();
    if( !pContext )
    {
            fprintf(stderr,"OpenCV. Problem to allocate memory for OpenCV context.");
        exit(1);
    }
    pthread_setspecific( g_TlsIndex, pContext );
    }
    return pContext;
#endif
#else /* CV_DLL */
    static CvContext* pContext = 0;

    if( !pContext )
    pContext = icvCreateContext();

    return pContext;
#endif
}


CV_IMPL CVStatus cvStdErrReport( CVStatus status, const char *funcName,
                                 const char *context, const char *file, int line )
{
    CvContext* cont = icvGetContext();

    if ( cvGetErrMode() == CV_ErrModeSilent )
    return ( status != CV_StsBackTrace ) ? ( cont->CVLastStatus = status ) : status;
    
    if (( status == CV_StsBackTrace ) || ( status == CV_StsAutoTrace ))
    fprintf(stderr, "\tcalled from ");
    else 
    {
    cont->CVLastStatus = status;
    fprintf(stderr, "OpenCV Error: %s \n\tin function ", cvErrorStr(status));
    }
    if ( line > 0 || file != NULL )
    fprintf(stderr,"[%s:%d]", file, line);
    fprintf(stderr,":%s", funcName ? funcName : "<unknown>");
    if ( context != NULL ) {
    if ( status != CV_StsAutoTrace )
        fprintf(stderr, "():\n%s", context);    /* Print context      */
    else
        fprintf(stderr, "(%s)", context);          /* Print arguments    */
    }
    fprintf(stderr, "\n");
    if ( cont->CVErrMode == CV_ErrModeLeaf ) {
    fprintf(stderr, "OpenCV: %s\n","terminating the application");
    exit(1);
    };

    return status;
}

CV_IMPL CVStatus cvGuiBoxReport( CVStatus status, const char *funcName, 
                 const char *context, const char *file, int line)
{

#ifdef WIN32

    char mess[1000];
    char title[100];
    char *choice = 0;
    const char* errText = cvErrorStr( status );


    if ( cvGetErrMode() != CV_ErrModeSilent )
    {
        if( !funcName ) funcName = "<unknown>";
        if( !context  ) context = "";
        if( !file     ) file = "";
        if(  line < 0 ) line = 0;

        if( cvGetErrMode() == CV_ErrModeLeaf )
            choice="\nErrMode=CV_ErrorModeLeaf\n"
                   "\nTerminate the application?";

        if( cvGetErrMode() == CV_ErrModeParent)
            choice="\nErrMode=CV_ErrorModeParent\n"
            "\nContinue?";

        if( status == CV_StsBackTrace)
            wsprintf( mess,"Called from %s(): [file %s, line %d]\n%s\n%s\n(status:%d)\n%s",
                      funcName, file,line,context, errText, status, choice);
        else if ( status == CV_StsAutoTrace )
            wsprintf( mess,"Called from %s(): [file %s, line %d]\n%s\n%s\n(status:%d)\n%s",
                      funcName, file, line, context, errText, status, choice);
        else
            wsprintf( mess,"In function %s(): [file %s, line %d]\n%s\n%s\n(status:%d)\n%s",
                      funcName, file, line, context,errText, status, choice);

        wsprintf(title,"OpenCV Beta 2: %s",cvErrorStr(cvGetErrStatus()));

        int answer = -1;

        if( (( cvGetErrMode()==CV_ErrModeParent) &&
            (IDCANCEL==MessageBox(NULL,mess,title,MB_ICONERROR|MB_OKCANCEL|MB_SYSTEMMODAL) ) ||
            ((cvGetErrMode() == CV_ErrModeLeaf) &&
            //(IDYES==MessageBox(NULL,mess,title,MB_ICONERROR|MB_YESNO|MB_SYSTEMMODAL))
            (IDABORT == (answer=MessageBox(NULL,mess,title,MB_ICONERROR|MB_ABORTRETRYIGNORE|MB_SYSTEMMODAL))||
            IDRETRY == answer)
            )))
        {
            if( answer == IDRETRY )
            {

    #if _MSC_VER >= 1200 || defined __ICL
                __asm int 3;
    #else
                assert(0);
    #endif
            }
            FatalAppExit(0,"OpenCV:\nterminating the application");
        }
    }

#else
    cvStdErrReport( status, funcName, context, file, line);
#endif

    return status;
}


CV_IMPL CVStatus cvNulDevReport( CVStatus status, const char *funcName,
                 const char *context, const char *file, int line)
{
  if( status||funcName||context||file||line )
  if ( cvGetErrMode() == CV_ErrModeLeaf )
      exit(1);
  return status;
}

CV_IMPL CVErrorCallBack cvRedirectError(CVErrorCallBack func)
{
    CvContext* context = icvGetContext();

    CVErrorCallBack old = context->CVErrorFunc;
    context->CVErrorFunc = func;
    return old;
} 
 
CV_IMPL const char* cvErrorStr(CVStatus status)
{
    static char buf[80];

    switch (status) 
    {
    case CV_StsOk :    return "No Error";
    case CV_StsBackTrace : return "Backtrace";
    case CV_StsError :     return "Unknown error";
    case CV_StsInternal :  return "Internal error";
    case CV_StsNoMem :     return "Insufficient memory";
    case CV_StsBadArg :    return "Bad argument";
    case CV_StsNoConv :    return "Iteration convergence failed";
    case CV_StsAutoTrace : return "Autotrace call";
    case CV_StsBadSize :   return "Bad/unsupported parameter of type CvSize";
    case CV_StsNullPtr :   return "Null pointer";
    case CV_StsDivByZero : return "Divizion by zero occured";
    case CV_BadStep :      return "Image step is wrong";
    case CV_StsInplaceNotSupported : return "Inplace operation is not supported";
    case CV_StsObjectNotFound :      return "Requested object was not found";
    case CV_BadDepth :     return "Input image depth is not supported by function";
    case CV_StsUnmatchedFormats : return "Formats of input arguments do not match"; 
    case CV_StsUnmatchedSizes :  return "Sizes of input arguments do not match";
    case CV_StsOutOfRange : return "One of arguments\' values is out of range";
    case CV_StsUnsupportedFormat : return "Unsupported format or combination of formats";
    case CV_BadCOI :      return "Input COI is not supported";
    case CV_BadNumChannels : return "Bad number of channels";
    case CV_StsBadFlag :   return "Bad flag (parameter or structure field)";
    case CV_StsBadPoint :  return "Bad parameter of type CvPoint";
    };

    sprintf(buf, "Unknown %s code %d", status >= 0 ? "status":"error", status);
    return buf;
}

CV_IMPL int cvGetErrMode(void)
{
    return icvGetContext()->CVErrMode;
}

CV_IMPL void cvSetErrMode( int mode )
{
    icvGetContext()->CVErrMode = mode;
}

CV_IMPL CVStatus cvGetErrStatus()
{
    return icvGetContext()->CVLastStatus;
}

CV_IMPL void cvSetErrStatus(CVStatus status)
{
    icvGetContext()->CVLastStatus = status;
}


/******************** Implementation of profiling stuff *********************/

/* initial assignment of profiling functions */
CvStartProfileFunc p_cvStartProfile = cvStartProfile;
CvEndProfileFunc p_cvEndProfile = cvEndProfile;


CV_IMPL void cvSetProfile( void (CV_CDECL *startprofile_f)(const char*, const char*, int),
               void (CV_CDECL *endprofile_f)(const char*, int))
{
    p_cvStartProfile = startprofile_f;
    p_cvEndProfile   = endprofile_f;
}

CV_IMPL void cvRemoveProfile()
{
    p_cvStartProfile = cvStartProfile;
    p_cvEndProfile   = cvEndProfile;
}

    

/* default implementation of cvStartProfile & cvEndProfile */
void CV_CDECL cvStartProfile(const char* call, const char* file, int line )
{   
#ifdef _CV_COMPILE_PROFILE_
    if( p_cvStartProfile != cvStartProfile )
    {
    p_cvStartProfile( call, file, line );
    }        
       
    /* default implementation */
    CvContext* context = icvGetContext();

    /* add record to stack */
    assert( context->CVStackCapacity >= context->CVStackSize ); 
    if( context->CVStackCapacity == context->CVStackSize )
    {
    /* increase stack */
    context->CVStackCapacity += 100;
    context->CVStack = (CvStackRecord*)realloc( context->CVStack, 
                      (context->CVStackCapacity) * sizeof(CvStackRecord) );
    }

    CvStackRecord* rec = &context->CVStack[context->CVStackSize];
    rec->file = file;
    rec->line = line;
    context->CVStackSize++;
#else 
    /* avoid warning "unreferenced value" */
    if( call||file||line) {}
    assert(0);
#endif
};

CV_IMPL void cvEndProfile( const char* file, int line )
{
#ifdef _CV_COMPILE_PROFILE_
    CvContext* context = icvGetContext();
    if( p_cvEndProfile != cvEndProfile )
    {
    p_cvEndProfile( file, line );
    }                
    /* default implementation */  
    context->CVStackSize--;

#else 
    /* avoid warning "unreferenced value" */
    if( file||line) {}
    assert(0);
#endif

};


CV_IMPL CVStatus cvError( CVStatus code, const char* funcName, 
              const char* msg, const char* file, int line )
{
    CvContext* context = icvGetContext();

    if ((code!=CV_StsBackTrace) && (code!=CV_StsAutoTrace))
    cvSetErrStatus(code);
    if (code == CV_StsOk)
    return CV_StsOk;
   
#ifdef _CV_COMPILE_PROFILE_

    int i;                                    
    char message[4096] = "";                              
    /* copy input message */                              
    strcpy( message, msg );                           
    /* append stack info */
    strcat( message, "\nStack\n{" );                               
    char* mes = message + strlen(message);

    for( i = 0; i < context->CVStackSize; i++ )                      
    {         
    i ? 0 : sprintf( mes,"\n" ), mes += strlen(mes); 
    CvStackRecord* rec = &(context->CVStack[i]);
    sprintf( mes, "   %s line %d\n", rec->file, rec->line ); 
    mes += strlen(mes);
    }
    strcat( message, "}\n" );

    context->CVErrorFunc( code, funcName, message, file, line );          
#else          
    context->CVErrorFunc( code, funcName, msg, file, line );

#endif
    return code;
};

CV_IMPL void cvGetCallStack(CvStackRecord** stack, int* size)
{
    CvContext* context = icvGetContext();
    *stack = context->CVStack;
    *size  = context->CVStackSize;
}

/******************** End of implementation of profiling stuff *********************/


/**********************DllMain********************************/

#ifdef CV_DLL

#ifdef WIN32
BOOL WINAPI DllMain( HINSTANCE /*hinstDLL*/,     /* DLL module handle        */
             DWORD     fdwReason,    /* reason called        */
             LPVOID    /*lpvReserved*/)  /* reserved             */
{
    CvContext *pContext;

    /// Note the actual size of the structure is larger.

    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:

    g_TlsIndex = TlsAlloc();
    if( g_TlsIndex == TLS_OUT_OF_INDEXES ) return FALSE;

    /* No break: Initialize the index for first thread. */
    /* The attached process creates a new thread. */

    case DLL_THREAD_ATTACH:

    pContext = icvCreateContext();
    if( pContext == NULL)
        return FALSE;
    TlsSetValue( g_TlsIndex, (LPVOID)pContext );
    break;

    case DLL_THREAD_DETACH:

    if( g_TlsIndex != TLS_OUT_OF_INDEXES ) 
    {
        pContext = (CvContext*)TlsGetValue( g_TlsIndex );
        if( pContext != NULL ) 
        {
        icvDestroyContext( pContext );
        }
    }
    break;

    case DLL_PROCESS_DETACH:

    if( g_TlsIndex != TLS_OUT_OF_INDEXES ) {
        pContext = (CvContext*)TlsGetValue( g_TlsIndex );
        if( pContext != NULL ) 
        {
        icvDestroyContext( pContext );
        }
        TlsFree( g_TlsIndex );
    }
    break;

    default:
    break;
    }
    return TRUE;
}
#else
/* POSIX pthread */

/* function - destructor of thread */
void icvPthreadDestructor(void* key_val)
{
    CvContext* context = (CvContext*) key_val;
    icvDestroyContext( context );
}

int pthrerr = pthread_key_create( &g_TlsIndex, icvPthreadDestructor );

#endif

#endif

/* function, which converts CvStatus to CVStatus */
IPCVAPI_IMPL( CVStatus,  icvErrorFromStatus, ( CvStatus status ) )
{
    switch (status) 
    {
    case CV_BADSIZE_ERR     : return CV_StsBadSize; //bad parameter of type CvSize
    case CV_NULLPTR_ERR     : return CV_StsNullPtr;
    case CV_DIV_BY_ZERO_ERR : return CV_StsDivByZero;
    case CV_BADSTEP_ERR     : return CV_BadStep ;
    case CV_OUTOFMEM_ERR    : return CV_StsNoMem;
    case CV_BADARG_ERR      : return CV_StsBadArg;
    case CV_NOTDEFINED_ERR  : return CV_StsError; //unknown/undefined err
    
    case CV_INPLACE_NOT_SUPPORTED_ERR: return CV_StsInplaceNotSupported;
    case CV_NOTFOUND_ERR : return CV_StsObjectNotFound;
    case CV_BADCONVERGENCE_ERR: return CV_StsNoConv;
    case CV_BADDEPTH_ERR     : return CV_BadDepth;
    case CV_UNMATCHED_FORMATS_ERR : return CV_StsUnmatchedFormats;

    case CV_UNSUPPORTED_COI_ERR      : return CV_BadCOI; 
    case CV_UNSUPPORTED_CHANNELS_ERR : return CV_BadNumChannels; 
    
    case CV_BADFLAG_ERR : return CV_StsBadFlag;//used when bad flag CV_ ..something
    
    case CV_BADRANGE_ERR    : return CV_StsBadArg; //used everywhere
    case CV_BADCOEF_ERR  :return CV_StsBadArg;     //used everywhere
    case CV_BADFACTOR_ERR:return CV_StsBadArg;     //used everywhere
    case CV_BADPOINT_ERR  :return CV_StsBadPoint;

    default: assert(0); return CV_StsError;
    }         
}         
/* End of file */


