%
% References:
%
% C. Lu. A Library of ADMM for Sparse and Low-rank Optimization. National University of Singapore, June 2016.
% https://github.com/canyilu/LibADMM.
% C. Lu, J. Feng, S. Yan, Z. Lin. A Unified Alternating Direction Method of Multipliers by Majorization 
% Minimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 527-541, 2018
%


addpath(genpath(cd))
clear

%% Examples for testing the low-rank matrix based models
% For detailed description of the sparse models, please refer to the Manual.


%% generate toy data
d = 10;
na = 200;
nb = 100;

A = randn(d,na);
X = randn(na,nb);
B = A*X;
b = B(:,1);

opts.tol = 1e-6; 
opts.max_iter = 1000;
opts.rho = 1.2;
opts.mu = 1e-3;
opts.max_mu = 1e10;
opts.DEBUG = 0;


%% RPCA
n1 = 100;
n2 = 200;
r = 10;
L = rand(n1,r)*rand(r,n2); % low-rank part

p = 0.1;
m = p*n1*n2;
temp = rand(n1*n2,1);
[~,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2);
Omega(I) = 1;
E = sign(rand(n1,n2)-0.5);
S = Omega.*E; % sparse part, S = P_Omega(E)

Xn = L+S;

lambda = 1/sqrt(max(n1,n2));
opts.loss = 'l1'; 
opts.DEBUG = 1;
tic
[Lhat,Shat,obj,err,iter] = rpca(Xn,lambda,opts);
toc
rel_err_L = norm(L-Lhat,'fro')/norm(L,'fro')
rel_err_S = norm(S-Shat,'fro')/norm(S,'fro')

err
iter


%% low rank matrix completion (lrmc) and regularized lrmc

n1 = 100;
n2 = 200;
r = 5;
X = rand(n1,r)*rand(r,n2);

p = 0.6;
omega = find(rand(n1,n2)<p);
M = zeros(n1,n2);
M(omega) = X(omega);
[Xhat,obj,err,iter] = lrmc(M, omega, opts);
rel_err_X = norm(Xhat-X,'fro')/norm(X,'fro')
 
E = randn(n1,n2)/100;
M = X+E;
lambda = 0.1;
[Xhat,obj,err,iter] = lrmcR(M, omega, lambda, opts);


%% low rank representation (lrr)
lambda = 0.001;
opts.loss = 'l21'; 
tic
[X,E,obj,err,iter] = lrr(A,A,lambda,opts);
toc
obj
err
iter

%% latent LRR (latlrr)
lambda = 0.1;
opts.loss = 'l1'; 
tic
[Z,L,obj,err,iter] = latlrr(A,lambda,opts);
toc
obj
err
iter

%% low rank and sparse representation (lrsr)
lambda1 = 0.1;
lambda2 = 4;
opts.loss = 'l21'; 
tic
[X,E,obj,err,iter] = lrsr(A,B,lambda1,lambda2,opts);
toc
obj
err
iter

%% improved graph clustering (igc)
n = 100;
r = 5;
X = rand(n,r)*rand(r,n);
C = rand(size(X));
lambda = 1/sqrt(n);
opts.loss = 'l1'; 
opts.DEBUG = 1;
tic
[L,S,obj,err,iter] = igc(X,C,lambda,opts);
toc
err
iter

%% multi-task low-rank affinity pursuit (mlap)
n1 = 100;
n2 = 200;
K = 10;
X = rand(n1,n2,K);
lambda = 0.1;
alpha = 0.2;
opts.loss = 'l1'; 
tic
[Z,E,obj,err,iter] = mlap(X,lambda,alpha,opts);
toc
err
iter

%% robust multi-view spectral clustering (rmsc)
n = 100;
r = 5;
m = 10;
X = rand(n,n,m);
lambda = 1/sqrt(n);
opts.loss = 'l1'; 
opts.DEBUG = 1;
tic
[L,S,obj,err,iter] = rmsc(X,lambda,opts);
toc
err
iter

%% sparse spectral clustering (sparsesc)
lambda = 0.001;
n = 100;
X = rand(n,n);
W = abs(X'*X);
I = eye(n);
D = diag(sum(W,1));
L = I - sqrt(inv(D))*W*sqrt(inv(D));
k = 5;
[P,obj,err,iter] = sparsesc(L,lambda,k,opts);
obj
err
iter


 


