function [X,obj,err,iter] = lrtr_Gaussian_tnn(A,b,Xsize,opts)

% Low tubal rank tensor recovery from Gaussian measurements by tensor
% nuclear norm minimization
%
% min_X ||X||_*, s.t. A*vec(X) = b
%
% ---------------------------------------------
% Input:
%       A       -    m*n matrix
%       b       -    m*1 vector
%       Xsize   -    Structure value in Matlab. The fields
%       (Xsize.n1,Xsize.n2,Xsize.n3) give the size of X.
%           
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    n1*n2*n3 tensor (n=n1*n2*n3)
%       obj     -    objective function value
%       err     -    residual
%       iter    -    number of iterations
%
% version 1.0 - 09/10/2017
%
% Written by Canyi Lu (canyilu@gmail.com)
%
% References:
% Canyi Lu, Jiashi Feng, Zhouchen Lin, Shuicheng Yan
% Exact Low Tubal Rank Tensor Recovery from Gaussian Measurements
% International Joint Conference on Artificial Intelligence (IJCAI). 2018


tol = 1e-8; 
max_iter = 1000;
rho = 1.1;
mu = 1e-6;
max_mu = 1e10;
DEBUG = 0;

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

n1 = Xsize.n1;
n2 = Xsize.n2;
n3 = Xsize.n3;
X = zeros(n1,n2,n3);
Z = X;
m = length(b);
Y1 = zeros(m,1);
Y2 = X;
I = eye(n1*n2*n3);
invA = (A'*A+I)\I;
iter = 0;
for iter = 1 : max_iter
    Xk = X;
    Zk = Z;
    % update X
    [X,Xtnn] = prox_tnn(Z-Y2/mu,1/mu);
    % update Z
    vecZ = invA*(A'*(-Y1/mu+b)+Y2(:)/mu+X(:));
    Z = reshape(vecZ,n1,n2,n3);
    
    dY1 = A*vecZ-b;
    dY2 = X-Z;
    chgX = max(abs(Xk(:)-X(:)));
    chgZ = max(abs(Zk(:)-Z(:)));
    chg = max([chgX chgZ max(abs(dY1)) max(abs(dY2(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            obj = Xtnn;
            err = norm(dY1)^2+norm(dY2(:))^2;
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y1 = Y1 + mu*dY1;
    Y2 = Y2 + mu*dY2;
    mu = min(rho*mu,max_mu);    
end
obj = Xtnn;
err = norm(dY1)^2+norm(dY2(:))^2;
