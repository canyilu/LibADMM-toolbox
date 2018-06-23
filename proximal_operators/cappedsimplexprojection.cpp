#include <iostream>
#include <vector>
#include <algorithm>
#include "mex.h"

using namespace std;

struct mypair
{
  double number;
  int index;
  
  void setval(double n, int i)
  {
    number=n;
    index=i;
  }
  
};

bool mycompare(mypair l, mypair r)
{
  return (l.number<r.number);
}

void cappedsimplexprojection(int N, double * y, double s, double * x, double * e)
{
  int i,j;
  
  if ((s<0)||(s>N)){
    cout<<"impossible sum constraint!\n"<<endl;
    exit(-1);
  }
  
  if (s==0){
    *e=0;
    for(i=0;i<N;i++){
      x[i]=0;
      (*e)+=y[i]*y[i];
    }
    (*e)*=0.5;
    return;
  }
  
  if (s==N){
    *e=0;
    for(i=0;i<N;i++){
      x[i]=1;
      (*e)+=(1-y[i])*(1-y[i]);
    }
    (*e)*=0.5;
    return;
  }
  
  // Sort y into ascending order.
  vector<mypair> v(N);
  for(i=0;i<N;i++){
    v[i].setval(y[i],i);
  }
  sort(v.begin(),v.end(),mycompare);
  
//   double T[N+1];  T[0]=0;
//   malloc(sizeof(double)*N)
//   double *T;
//   T=(double*)malloc(N+1);
//   T[0]=0;

  double *T;
  T = new double[N+1];
  T[0]=0;

      

  // Compute partial sums.
  for(i=1; i<=N; i++) T[i]=T[i-1]+v[i-1].number;
  
  double gamma;
  // i is the number of 0's in the solution.
  // j is the number of (0,1)'s in the solution.
  bool flag=false;
  for(i=0;i<=N;i++){
    
    // i==j
    if ((i+s)==N)
      if((i==0) || (v[i].number>=v[i-1].number+1)){
        j=i;
        flag=true;
        break;
      }
    
    // i<j
    for(j=i+1;j<=N;j++){
      gamma=(s+j-N+T[i]-T[j])/(j-i);
      //cout<<"gamma="<<gamma<<endl;
       
      if (i==0)
        if (j==N) {
          if ((v[i].number+gamma>0) && (v[j-1].number+gamma<1)) {flag=true;break;}
        }
        else {
          if ((v[i].number+gamma>0) && (v[j-1].number+gamma<1) && (v[j].number+gamma>=1)) {flag=true; break;}
        }
      else
        if (j==N) {
          if ((v[i-1].number+gamma<=0) && (v[i].number+gamma>0) && (v[j-1].number+gamma<1)) {flag=true;break;}
        }
        else {
          if ((v[i-1].number+gamma<=0) && (v[i].number+gamma>0) && (v[j-1].number+gamma<1) && (v[j].number+gamma>=1)) {flag=true;break;}
        }
    }
    
    if(flag) break;
  }
  
  // get the solution in original order.
  *e=0;
  int k;
  for(k=0;k<i;k++){
    x[v[k].index]=0;
    (*e)+=(v[k].number)*(v[k].number);
  }
  
  for(k=i;k<j;k++){
    x[v[k].index]=v[k].number+gamma;
    (*e)+=gamma*gamma;
  }
  
  for(k=j;k<N;k++){
    x[v[k].index]=1;
    (*e)+=(1-v[k].number)*(1-v[k].number);
  }
  
  (*e)*=0.5;
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
  
  /* check for proper number of arguments */
  if(nrhs!=2)
    mexErrMsgIdAndTxt("projection:invalidNumInputs", "Two inputs (y,s) required.");
  
  int M=mxGetM(prhs[0]);
  int N=mxGetN(prhs[0]);
  if((M!=1)&&(N!=1))
    mexErrMsgIdAndTxt("projection:invalidDimensions", "First argument y needs to be vector.");
  int Length=(N>M)?N:M;
  
  plhs[0] = mxCreateDoubleMatrix((mwSize)M, (mwSize)N, mxREAL);
  plhs[1] = mxCreateDoubleMatrix((mwSize)1, (mwSize)1, mxREAL);
  
  double * y=mxGetPr(prhs[0]);
  double s=mxGetScalar(prhs[1]);
  double * x=mxGetPr(plhs[0]);
  double * e =mxGetPr(plhs[1]);
  
  cappedsimplexprojection(Length, y, s, x, e);
}

/*
 * int main(int argc,char * argv[])
 * {
 *
 * int N=6;
 *
 * double y[6]={0.5377,    1.8339,    -2.2588,    0.8622,    0.3188,   -1.3077};
 * double s=10;
 * double d[6]={0.2785,    0.5469,    0.9575,    0.9649,    0.1576,    0.9706};
 *
 * double x[6];
 * double alpha;
 *
 * cappedsimplexprojection(N, y, s, d, x, &alpha);
 *
 * cout<<alpha<<endl;
 *
 * for(int i=0;i<N;i++)
 * cout<<x[i]<<"   ";
 * cout<<endl;
 * }
 */



