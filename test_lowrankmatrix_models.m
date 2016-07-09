addpath(genpath(cd))
clear
%% data
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
opts.DEBUG = 1;


%% test LRR
lambda = 0.001;
opts.loss = 'l21'; 
tic
[X,E,obj,err,iter] = lrr(A,A,lambda,opts);
toc
obj
err
iter

%% test Low-Rank and Sparse Representation
lambda1 = 0.1;
lambda2 = 4;
opts.loss = 'l21'; 
tic
[X,E,obj,err,iter] = lrsr(A,B,lambda1,lambda2,opts);
toc
obj
err
iter

%% test Latent LRR
lambda = 0.1;
opts.loss = 'l1'; 
tic
[Z,L,obj,err,iter] = latlrr(A,lambda,opts);
toc
obj
err
iter

%% test RPCA
n = 100;
r = 5;
X = rand(n,r)*rand(r,n);
lambda = 1/sqrt(n);
opts.loss = 'l1'; 
opts.DEBUG = 1;
tic
[L,S,obj,err,iter] = rpca(X,lambda,opts);
toc
recoveryerror = norm(L-X,'fro')
err
iter

%% test IGC, improved graph clustering
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

%% test Multi-task Low-rank Affinity Pursuit
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

%% test RMSC
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

%% test sparse spectral clustering
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

%% test LRMC and LRMCR

n1 = 100;
n2 = 200;
r = 5;
X = rand(n1,r)*rand(r,n2);

p = 0.6;
omega = find(rand(n1,n2)<p);
M = zeros(n1,n2);
M(omega) = X(omega);
[Xhat,obj,err,iter] = lrmc(M, omega, opts);
norm(Xhat-X,'fro')
 
E = randn(n1,n2)/100;
M = X+E;
lambda = 0.1;
[Xhat,obj,err,iter] = lrmcR(M, omega, lambda, opts);
norm(Xhat-X,'fro')
 


