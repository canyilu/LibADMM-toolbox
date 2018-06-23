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

%% Examples for testing the low-rank tensor models
% For detailed description of the sparse models, please refer to the Manual.


opts.mu = 1e-6;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 1;


%% Tensor RRPCA based on sum of nuclear norm minimization (rpca_snn)
n1 = 50;
n2 = n1;
n3 = n1;
r = 5
L = rand(r,r,r);
U1 = rand(n1,r);
U2 = rand(n2,r);
U3 = rand(n3,r);
L = nmodeproduct(L,U1,1);
L = nmodeproduct(L,U2,2);
L = nmodeproduct(L,U3,3); % low rank part

p = 0.05;
m = p*n1*n2*n3;
temp = rand(n1*n2*n3,1);
[~,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2,n3);
Omega(I) = 1;
E = sign(rand(n1,n2,n3)-0.5);
S = Omega.*E; % sparse part, S = P_Omega(E)

Xn = L+S;

lambda = sqrt([max(n1,n2*n3), max(n2,n1*n3), max(n3,n1*n2)]);
lambda = [1 1 1]
[Lhat,Shat,err,iter] = trpca_snn(Xn,lambda,opts);

err
iter


%% low-rank tensor completion based on sum of nuclear norm minimization (lrtc_snn) 
n1 = 50;
n2 = n1;
n3 = n1;
r = 5;
X = rand(r,r,r);
U1 = rand(n1,r);
U2 = rand(n2,r);
U3 = rand(n3,r);
X = nmodeproduct(X,U1,1);
X = nmodeproduct(X,U2,2);
X = nmodeproduct(X,U3,3);
p = 0.5;
omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);

lambda = [1 1 1];
[Xhat,err,iter] = lrtc_snn(M,omega,lambda,opts);
err
iter
RSE = norm(X(:)-Xhat(:))/norm(X(:))

%% regularized low-rank tensor completion based on sum of nuclear norm minimization (lrtcR_snn)
n1 = 50;
n2 = n1;
n3 = n1;
r = 5;
X = rand(r,r,r);
U1 = rand(n1,r);
U2 = rand(n2,r);
U3 = rand(n3,r);
X = nmodeproduct(X,U1,1);
X = nmodeproduct(X,U2,2);
X = nmodeproduct(X,U3,3);
p = 0.5;
omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);
lambda = [1 1 1];
[Xhat,err,iter] = lrtcR_snn(M,omega,lambda,opts);
err
iter


%% Tensor RRPCA based on tensor nuclear norm minimization (rpca_tnn)
n1 = 50;
n2 = n1;
n3 = n1;
r = 0.1*n1 % tubal rank
L1 = randn(n1,r,n3)/n1;
L2 = randn(r,n2,n3)/n2;
L = tprod(L1,L2); % low rank part

p = 0.1;
m = p*n1*n2*n3;
temp = rand(n1*n2*n3,1);
[~,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2,n3);
Omega(I) = 1;
E = sign(rand(n1,n2,n3)-0.5);
S = Omega.*E; % sparse part, S = P_Omega(E)

Xn = L+S;
lambda = 1/sqrt(n3*max(n1,n2));

tic
[Lhat,Shat] = trpca_tnn(Xn,lambda,opts);

RES_L = norm(L(:)-Lhat(:))/norm(L(:))
RES_S = norm(S(:)-Shat(:))/norm(S(:))
trank = tubalrank(Lhat)



%% low-rank tensor completion based on tensor nuclear norm minimization (lrtc_tnn)
n1 = 50;
n2 = n1;
n3 = n1;
r = 0.1*n1 % tubal rank
L1 = randn(n1,r,n3)/n1;
L2 = randn(r,n2,n3)/n2;
X = tprod(L1,L2); % low rank part
p = 0.5;
omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);

[Xhat,obj,err,iter] = lrtc_tnn(M,omega,opts);

err
iter
RSE = norm(X(:)-Xhat(:))/norm(X(:))
trank = tubalrank(Xhat)



%% regularized low-rank tensor completion based on tensor nuclear norm minimization (lrtcR_tnn) 
n1 = 50;
n2 = n1;
n3 = n1;
r = 0.1*n1 % tubal rank
L1 = randn(n1,r,n3)/n1;
L2 = randn(r,n2,n3)/n2;
X = tprod(L1,L2); % low rank part
p = 0.5;
omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);

lambda = 0.5;
[Xhat,Ehat,obj,err,iter] = lrtcR_tnn(M,omega,lambda,opts);
err
iter


%% low-rank tensor recovery from Gaussian measurements based on tensor nuclear norm minimization (lrtr_Gaussian_tnn)
n1 = 30;
n2 = n1; 
n3 = 5;
r = 0.2*n1; % tubal rank
X = tprod(randn(n1,r,n3),randn(r,n2,n3)); % size: n1*n2*n3

m = 3*r*(n1+n2-r)*n3+1; % number of measurements
n = n1*n2*n3;
A = randn(m,n)/sqrt(m);

b = A*X(:);
Xsize.n1 = n1;
Xsize.n2 = n2;
Xsize.n3 = n3;

opts.DEBUG = 1;
[Xhat,obj,err,iter]  = lrtr_Gaussian_tnn(A,b,Xsize,opts);

RSE = norm(Xhat(:)-X(:))/norm(X(:))
trank = tubalrank(Xhat)

