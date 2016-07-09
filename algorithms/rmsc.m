function [L,S,obj,err,iter] = rmsc(X,lambda,opts)

% Solve the Robust Multi-view Spectral Clustering (RMSC) problem by M-ADMM
%
% min_{L,S_i} ||L||_*+lambda*\sum_i ||S_i||_1,
% s.t. X_i=L+S_i, i=1,...,m, L>=0, L1=1.
% ---------------------------------------------
% Input:
%       X       -    d*n*m tensor
%       lambda  -    >0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       L       -    d*n matrix
%       S       -    d*n*m tensor
%       obj     -    objective function value
%       err     -    residual
%       iter    -    number of iterations
%
% version 1.0 - 19/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
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

[d,n,m] = size(X);
L = zeros(d,n);
S = zeros(d,n,m);
Z = L;
Y = S;
dY = S;
Y2 = L;
iter = 0;
for iter = 1 : max_iter
    Lk = L;
    Sk = S;
    Zk = Z;
    % first super block {Z,S_i}
    [Z,nuclearnormZ] = prox_nuclear(L+Y2/mu,1/mu);
    for i = 1 : m
        S(:,:,i) = prox_l1(-L+X(:,:,i)-Y(:,:,i)/mu,lambda/mu);
    end
    % second super block {L}
    temp = (sum(X-S-Y/mu,3)+Z-Y2/mu)/(m+1);
    L = project_simplex(temp);

    for i = 1 : m
        dY(:,:,i) = L+S(:,:,i)-X(:,:,i);
    end
    dY2 = L-Z;
    chgL = max(abs(Lk(:)-L(:)));
    chgZ = max(abs(Zk(:)-Z(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([chgL chgS chgZ max(abs(dY(:))) max(abs(dY2(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            obj = nuclearnormZ+lambda*norm(S(:),1);
            err = sqrt(norm(dY(:))^2+norm(dY2,'fro')^2);
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    Y2 = Y2 + mu*dY2;
    mu = min(rho*mu,max_mu);    
end
obj = nuclearnormZ+lambda*norm(S(:),1);
err = sqrt(norm(dY(:))^2+norm(dY2,'fro')^2);

