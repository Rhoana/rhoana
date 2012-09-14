/* file:        mexutils.c 
** author:      Andrea Vedaldi
** description: Utility functions to write MEX files.
**/

#include"mex.h"
#include<math.h>

#undef M_PI
#define M_PI 3.14159265358979

/** @biref Is real scalar?
 **
 ** @return @c true if the array @a A is a real scalar.
 **/
int
uIsRealScalar(const mxArray* A)
{
  return 
    mxIsDouble(A) && 
    !mxIsComplex(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    mxGetM(A) == 1 &&
    mxGetN(A) == 1 ;
}

/** @brief Is real matrix?
 **
 ** The function checks wether the argument @a A is a real matrix.  In
 ** addition, if @a M >= 0, it checks wether the number of rows is
 ** equal to @a M and, if @a N >= 0, if the number of columns is equal
 ** to @a N.
 **
 ** @param M number of rows.
 ** @param N number of columns.
 ** @return @c true if the array is a real matrix with the specified format.
 **/
int
uIsRealMatrix(const mxArray* A, int M, int N)
{
  return  
    mxIsDouble(A) &&
    !mxIsComplex(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    ((M>=0)?(mxGetM(A) == M):1) &&
    ((N>=0)?(mxGetN(A) == N):1) ;   
}

/** @brief Is real vector?
 **
 ** The function checks wether the argument  @a V is a real vector. By
 ** definiton, a  matrix is a vector  if one of its  dimension is one.
 ** In addition, if  @a D >= 0, it checks wether  the dimension of the
 ** vecotr is equal to @a D.
 **
 ** @param D lenght of the vector.
 ** @return @c true if the array is a real vector of the specified dimension.
 **/
int
uIsRealVector(const mxArray* V, int D) 
{
  int M = mxGetM(V) ;
  int N = mxGetN(V) ;
  int is_vector = (N == 1) || (M == 1) ;
  
  return   
    mxIsDouble(V) &&
    !mxIsComplex(V) &&
    mxGetNumberOfDimensions(V) == 2 &&
    is_vector &&
    ( D < 0 || N == D || M == D) ;
}


/** @brief Is a string?
 **
 ** The function checks wether the array @a S is a string. If
 ** @a L is non-negative, it also check wether the strign has
 ** length @a L.
 **
 ** @return @a c true if S is a string of the specified length.
 **/
int
uIsString(const mxArray* S, int L)
{
  int M = mxGetM(S) ;
  int N = mxGetN(S) ;

  return 
    mxIsChar(S) &&
    M == 1 &&
    (L < 0 || N == L) ;
}

/**
 **
 **/

