#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"

#include "flsa.h"


/*

  Functions contained in "flsa.h"

1. The algorithm for sloving (1) with a given (labmda1, lambda2)
 
  void flsa(double *x, double *z, double *info,
		  double * v, double *z0, 
		  double lambda1, double lambda2, int n, 
		  int maxStep, double tol, int tau, int flag)
*/


/*

  We solve the Fused Lasso Signal Approximator (FLSA) problem:

     min_x  1/2 \|x-v\|^2  + lambda1 * \|x\|_1 + lambda2 * \|A x\|_1,      (1)

  It can be shown that, if x* is the solution to

     min_x  1/2 \|x-v\|^2  + lambda2 \|A x\|_1,                            (2)

  then 
     x**= sgn(x*) max(|x*|-lambda_1, 0)                                    (3)

  is the solution to (1).

  By some derivation (see the description in sfa.h), (2) can be solved by

     x*= v - A^T z*,

  where z* is the optimal solution to

     min_z  1/2  z^T A AT z - < z, A v>,
		subject to  \|z\|_{infty} \leq lambda2                             (4)
*/



/*


  In flsa, we solve (1) corresponding to a given (lambda1, lambda2)

  void flsa(double *x, double *z, double *gap,
		  double * v, double *z0, 
		  double lambda1, double lambda2, int n, 
		  int maxStep, double tol, int flag)

  Output parameters:
      x:        the solution to problem (1)
	  z:        the solution to problem (4)
	  infor:    the information about running the subgradient finding algorithm
	                 infor[0] = gap:         the computed gap (either the duality gap
	                                            or the summation of the absolute change of the adjacent solutions)
					 infor[1] = steps:       the number of iterations
					 infor[2] = lambad2_max: the maximal value of lambda2_max
					 infor[3] = numS:        the number of elements in the support set
								
  Input parameters:
      v:        the input vector to be projected
	  z0:       a guess of the solution of z

	  lambad1:  the regularization parameter
	  labmda2:  the regularization parameter
	  n:        the length of v and x

      maxStep:  the maximal allowed iteration steps
	  tol:      the tolerance parameter
	  flag:     the flag for initialization and deciding calling sfa
                     switch (flag)
					     >0: sfa
						 <0: sfa_ls

                     switch ( abs(flag))
					     case 1, 2, 3, or 4: 
						               z0 is a "good" starting point 
						               (such as the warm-start of the previous solution,
									   or the user want to test the performance of this starting point;
									   the starting point shall be further projected to the L_{infty} ball,
									   to make sure that it is feasible)

						 case 11, 12, 13, or 14: z0 is a "random" guess, and thus not used
						               (we shall initialize z with zero if lambda2 is less than 0.5 *zMax
									         and otherwise initialize z with zero with the solution of the linear system;
											 this solution is projected to the L_{infty} ball)

*/


/*

We write the wrapper for calling from Matlab

void flsa(double *x, double *z, double *gap,
		  double * v, double *z0, 
		  double lambda1, double lambda2, int n, 
		  int maxStep, double tol, int flag)
*/


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* v=            mxGetPr(prhs[0]);
	double* z0=           mxGetPr(prhs[1]);

	double lambda1=       mxGetScalar(prhs[2]);
	double lambda2=       mxGetScalar(prhs[3]);
    int     n=   (int )   mxGetScalar(prhs[4]);

	int    maxStep= (int) mxGetScalar(prhs[5]);
	double tol=           mxGetScalar(prhs[6]);
	int    tau=     (int) mxGetScalar(prhs[7]);
	int    flag= (int)    mxGetScalar(prhs[8]);
	
    
    double *x, *z, *infor;
    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix( n, 1, mxREAL); 	
    plhs[1] = mxCreateDoubleMatrix( n-1, 1, mxREAL); 
	plhs[2] = mxCreateDoubleMatrix( 1, 4, mxREAL);
    x=  mxGetPr(plhs[0]);
	z=  mxGetPr(plhs[1]);
	infor=mxGetPr(plhs[2]);

	flsa(x, z, infor,
		  v, z0, 
		  lambda1, lambda2, n, 
		  maxStep, tol, tau, flag);
}

