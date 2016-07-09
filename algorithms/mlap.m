function [Z,E,obj,err,iter] = mlap(X,lambda,alpha,opts)

% Solve the Multi-task Low-rank Affinity Pursuit (MLAP) minimization problem by M-ADMM
%
% Reference: Cheng, Bin, Guangcan Liu, Jingdong Wang, Zhongyang Huang, and Shuicheng Yan.
% Multi-task low-rank affinity pursuit for image segmentation. ICCV, 2011.
%
% min_{Z_i,E_i} \sum_{i=1}^K (||Z_i||_*+lambda*loss(E_i))+alpha*||Z||_{2,1}, 
% s.t. X_i=X_i*Z_i+E_i, i=1,...,K.
% loss(E) = ||E||_1 or 0.5*||E||_F^2 or ||E||_{2,1}
%
% ---------------------------------------------
% Input:
%       X       -    d*n*K tensor
%       lambda  -    >0, parameter
%       alpha   -    >0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.loss       -   'l1': loss(E) = ||E||_1 
%                               'l2': loss(E) = 0.5*||E||_F^2
%                               'l21' (default): loss(E) = ||E||_{2,1}
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       Z       -    n*n*K tensor
%       E       -    d*n*K tensor
%       obj     -    objective function value
%       err     -    residual
%       iter    -    number of iterations
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;
DEBUG = 0;
loss = 'l21';

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'loss');        loss = opts.loss;            end
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

[d,n,K] = size(X);
Z = zeros(n,n,K);
E = zeros(d,n,K);
J = Z;
S = Z;
Y = E;
W = Z;
V = Z;
dY = Y;
XmXS = E;
XtX = zeros(n,n,K);
invXtXI = zeros(n,n,K);
I = eye(n);
for i = 1 : K
    XtX(:,:,i) = X(:,:,i)'*X(:,:,i);
    invXtXI(:,:,i) = (XtX(:,:,i)+I)\I;
end
nuclearnormJ = zeros(K,1);

iter = 0;
for iter = 1 : max_iter
    Zk = Z;
    Ek = E;
    Jk = J;
    Sk = S;
    % first super block {J,S}
    for i = 1 : K
        [J(:,:,i),nuclearnormJ(i)] = prox_nuclear(Z(:,:,i)+W(:,:,i)/mu,1/mu);
        S(:,:,i) = invXtXI(:,:,i)*(XtX(:,:,i)-X(:,:,i)'*(E(:,:,i)-Y(:,:,i)/mu)+Z(:,:,i)+(V(:,:,i)-W(:,:,i))/mu);
    end
    % second super block {Z,E}
    Z = prox_tensor_l21((J+S-(W+V)/mu)/2,alpha/(2*mu));
    for i = 1 : K
        XmXS(:,:,i) = X(:,:,i)-X(:,:,i)*S(:,:,i);
    end
    if strcmp(loss,'l1')
        for i = 1 : K
            E(:,:,i) = prox_l1(XmXS(:,:,i)+Y(:,:,i)/mu,lambda/mu);
        end
    elseif strcmp(loss,'l21')
        for i = 1 : K
            E(:,:,i) = prox_l21(XmXS(:,:,i)+Y(:,:,i)/mu,lambda/mu);
        end
    elseif strcmp(loss,'l2')
        for i = 1 : K
            E = (XmXS(:,:,i)+Y(:,:,i)/mu) / (lambda/mu+1);
        end        
    else
        error('not supported loss function');
    end
    
    dY = XmXS-E;
    dW = Z-J;
    dV = Z-S;

    chgZ = max(abs(Zk(:)-Z(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chgJ = max(abs(Jk(:)-J(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([chgZ chgE chgJ chgS max(abs(dY(:))) max(abs(dW(:))) max(abs(dV(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            obj = sum(nuclearnormJ)+lambda*comp_loss(E,loss)+alpha*comp_loss(Z,'l21');
            err = sqrt(norm(dY(:))^2+norm(dW(:))^2+norm(dV(:))^2);
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    W = W + mu*dW;
    V = V + mu*dV;
    mu = min(rho*mu,max_mu);    
end
obj = sum(nuclearnormJ)+lambda*comp_loss(E,loss)+alpha*comp_loss(Z,'l21');
err = sqrt(norm(dY(:))^2+norm(dW(:))^2+norm(dV(:))^2);
 
function X = prox_tensor_l21(B,lambda)
% proximal operator of tensor l21-norm, i.e., the sum of the l2 norm of all
% tubes of a tensor. 
% 
% X     -   n1*n2*n3 tensor
% B     -   n1*n2*n3 tensor
% 
% min_X lambda*\sum_{i=1}^n1\sum_{j=1}^n2 ||X(i,j,:)||_2 + 0.5*||X-B||_F^2

[n1,n2,n3] = size(B);
X = zeros(n1,n2,n3);
for i = 1 : n1
    for j = 1 : n2
        v = B(i,j,:);
        nxi = norm(v(:));
        if nxi > lambda
            X(i,j,:) = (1-lambda/nxi)*B(i,j,:);
        end        
    end
end
