#include "mex.h"
#include "blas.h"
/*
  dk = dChol2(HL, Ldd, Knd, Kdd, dKnd, dKdd)
 
  derivative of pairwise HSIC for kernels K and L
  HL:       column-centered n x d submatrix of L
  Ldd, Kdd: inverse d x d submatrices
  Knd:      n x d submatrix of K
  dKnd:     d(Knd) 
  dKdd:     derivative of ddimK x dddimK submatrix (both vec'ed)

  computes
  2 * vec(HL*Ldd*HL' * Knd * Kdd)' * dKnd 
  - vec(Kdd*Knd* HL*Ldd*HL' *Knd'*Kdd) * dKdd

*/

/*
***** BEGIN LICENSE BLOCK *****
Version: MPL 1.1/GPL 2.0/LGPL 2.1

The contents of this file are subject to the Mozilla Public License Version
1.1 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.mozilla.org/MPL/

Software distributed under the License is distributed on an "AS IS" basis,
WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
for the specific language governing rights and limitations under the
License.

The Original Code is Fast Kernel Independent Component Analysis using
an Approximate Newton Method.

The Initial Developers of the Original Code are
Stefanie Jegelka, Hao Shen,  Arthur Gretton, and Francis Bach.
Portions created by the Initial Developers are Copyright (C) 2007
the Initial Developers. All Rights Reserved.

Contributors:
Stefanie Jegelka,
Hao Shen,
Arthur Gretton,
Francis Bach

Alternatively, the contents of this file may be used under the terms of
either the GNU General Public License Version 2 or later (the "GPL"), or
the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
in which case the provisions of the GPL or the LGPL are applicable instead
of those above. If you wish to allow use of your version of this file only
under the terms of either the GPL or the LGPL, and not to allow others to
use your version of this file under the terms of the MPL, indicate your
decision by deleting the provisions above and replace them with the notice
and other provisions required by the GPL or the LGPL. If you do not delete
the provisions above, a recipient may use your version of this file under
the terms of any one of the MPL, the GPL or the LGPL.

***** END LICENSE BLOCK *****
*/

// #ifdef _WIN32
//   double ddot(int*, double*, int*, double*, int*);
// #else
//   double ddot_(int*, double*, int*, double*, int*);
// #endif

// #if !defined(_WIN32)
// #define ddot ddot_
// #endif

void mexFunction(int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[])
{ 
  /* matrices */
  double *hl, *y, *ldd, *kdd, *knd, *dkdd, *dknd, *tmp, *tmp1, *tmp3, *res;
  
  /* n: number of samples, 
     ddimK, ddimL: number of cholesky indices, 
     m: number of sources */
  mwSize n, ddimK, ddimL, m;
  
  /* for multiplication */
  mwSize oneI = 1, nd;
  double nullD = 0.0, oneD = 1.0, mOneD = -1.0, twoD = 2.0;
  char *chn = "N", *cht = "T";

  hl = mxGetPr(prhs[0]);
  ldd = mxGetPr(prhs[1]);
  knd = mxGetPr(prhs[2]);
  kdd = mxGetPr(prhs[3]);
  dknd = mxGetPr(prhs[4]);
  dkdd = mxGetPr(prhs[5]);

  ddimK = mxGetN(prhs[3]);
  ddimL = mxGetN(prhs[1]);
  n= mxGetM(prhs[2]);
  m = mxGetN(prhs[4]);
  nd = n*ddimK;

     
  tmp = mxCalloc(ddimL*ddimK, sizeof(double));  
  y = mxCalloc(ddimL*ddimK, sizeof(double));
  
  plhs[0] = mxCreateDoubleMatrix(1,m, mxREAL);
  res = mxGetPr(plhs[0]);

  /* y = (hl'*knd) * kdd   (ddimL x ddimK) */
  dgemm_(cht, chn, (const long *)&ddimL, (const long *)&ddimK, (const long *)&n, &oneD, hl, (const long *)&n, knd, (const long *)&n, &nullD, tmp, (const long *)&ddimL);
  dgemm_(chn, chn, (const long *)&ddimL, (const long *)&ddimK, (const long *)&ddimK, &oneD, tmp, (const long *)&ddimL, kdd, (const long *)&ddimK, &nullD, y, (const long *)&ddimL);

  /* tmp = Ldd * y  (ddimL * ddimK) */
  dgemm_(chn, chn, (const long *)&ddimL, (const long *)&ddimK, (const long *)&ddimL, &oneD, ldd, (const long *)&ddimL, y, (const long *)&ddimL, &nullD, tmp, (const long *)&ddimL);

  /* term1: vec( hl * tmp)' * dKnd */
  tmp1 = mxCalloc(n*ddimK, sizeof(double));
  dgemm_(chn, chn, (const long *)&n, (const long *)&ddimK, (const long *)&ddimL, &oneD, hl, (const long *)&n, tmp, (const long *)&ddimL, &nullD, tmp1, (const long *)&n);
  dgemm_(cht, chn, (const long *)&oneI, (const long *)&m, (const long *)&nd, &oneD, tmp1, (const long *)&nd, dknd, (const long *)&nd, &nullD, res, (const long *)&oneI);

  /* term2: vec( y' * tmp)' * dKdd 
     result is 2*term1 - term2 */
  nd = ddimK * ddimK;
  mxFree(tmp1);
  tmp1 = mxCalloc(nd,sizeof(double));
  dgemm_(cht, chn, (const long *)&ddimK, (const long *)&ddimK, (const long *)&ddimL, &oneD, y, (const long *)&ddimL, tmp, (const long *)&ddimL, &nullD, tmp1, (const long *)&ddimK);
  mxFree(tmp);
  mxFree(y); 

  dgemm_(cht, chn, (const long *)&oneI, (const long *)&m, (const long *)&nd, &mOneD, tmp1, (const long *)&nd, dkdd, (const long *)&nd, &twoD, res, (const long *)&oneI);

  mxFree(tmp1);

  return;
}
