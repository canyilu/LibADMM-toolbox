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

%% Examples for testing the sparse models
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
opts.rho = 1.1;
opts.mu = 1e-4;
opts.max_mu = 1e10;
opts.DEBUG = 0;

%% l1
[X2,obj,err,iter] = l1(A,B,opts);
iter
obj
err
stem(X2(:,1))

%% group l1
g_num = 5;
g_len = round(na/g_num);
for i = 1 : g_num-1
    G{i} = (i-1)*g_len+1 : i*g_len;
end
G{g_num} = (g_num-1)*g_len+1:na;

[X2,obj,err,iter] = groupl1(A,B,G,opts);
iter
obj
err
stem(X2(:,1))

%% elastic net
lambda = 0.01;
[X2,obj,err,iter] = elasticnet(A,B,lambda,opts);
iter
obj
err
stem(X2(:,1))

%% fused Lasso
lambda = 0.01;
[x,obj,err,iter] = fusedl1(A,b,lambda,opts);
iter
obj
err
stem(x)

%% trace Lasso
[x,obj,err,iter] = tracelasso(A,b,opts);
iter
obj
err
stem(x)

%% k-support norm 
k = 10;
[X,err,iter] = ksupport(A,B,k,opts);
iter
err
stem(X(:,1));

%% --------------------------------------------------------------

%% regularized l1
lambda = 0.01;
opts.loss = 'l1'; 
[X,E,obj,err,iter] = l1R(A,B,lambda,opts);
iter
obj
err
stem(X(:,1)) 

%% regularized group Lasso
g_num = 5;
g_len = round(na/g_num);

for i = 1 : g_num-1
    G{i} = (i-1)*g_len+1 : i*g_len;
end
G{g_num} = (g_num-1)*g_len+1:na;
lambda = 1;
opts.loss = 'l1'; 
[X,E,obj,err,iter] = groupl1R(A,B,G,lambda,opts);
iter
obj
err
stem(X(:,1))
 
%% regularized elastic net
lambda1 = 10;
lambda2 = 10;
opts.loss = 'l1'; 
[X,E,obj,err,iter] = elasticnetR(A,B,lambda1,lambda2,opts);
iter
obj
err
stem(X(:,1))
% stem(E(:,1))

%% regularized fused Lasso
lambda1 = 10;
lambda2 = 10;
opts.loss = 'l1';
[X,E,obj,err,iter] = fusedl1R(A,b,lambda1,lambda2,opts);
iter
obj
err
stem(X(:,1))
stem(E(:,1))


%% regularized trace Lasso
lambda = 0.1;
opts.loss = 'l1'; 
tic
[x,e,obj,err,iter] = tracelassoR(A,b,lambda,opts);
toc
iter
obj
err
stem(x)

%% regularized k-support norm
lambda = 0.1;
k = 10;
[X,E,err,iter] = ksupportR(A,B,lambda,k,opts);
iter
err
stem(X(:,1));

