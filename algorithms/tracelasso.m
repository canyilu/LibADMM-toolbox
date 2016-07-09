function [x,obj,err,iter] = tracelasso(A,b,opts)

% Solve the trace Lasso minimization problem by ADMM
%
% min_x ||A*Diag(x)||_*, s.t. Ax=b
%
% ---------------------------------------------
% Input:
%       A       -    d*n matrix
%       b       -    d*1 vector
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       x       -    n*1 vector
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

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

[d,n] = size(A);
x = zeros(n,1);
Z = zeros(d,n);
Y1 = zeros(d,1);
Y2 = Z;
Atb = A'*b;
AtA = A'*A;
invAtA = (AtA+diag(diag(AtA)))\eye(n);

iter = 0;
for iter = 1 : max_iter
    xk = x;
    Zk = Z;
    % update x
    x = invAtA*(-A'*Y1/mu+Atb+diagAtB(A,-Y2/mu+Z));
    % update Z
    [Z,nuclearnorm] = prox_nuclear(A*diag(x)+Y2/mu,1/mu);

    dY1 = A*x-b;
    dY2 = A*diag(x)-Z;
    chgx = max(abs(xk-x));
    chgZ = max(abs(Zk-Z));
    chg = max([chgx chgZ max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            obj = nuclearnorm;
            err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);
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
obj = nuclearnorm;
err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);

function v = diagAtB(A,B)
% A, B - d*n matrices
% v = diag(A'*B), n*1 vector

n = size(A,2);
v = zeros(n,1);
for i = 1 : n
   v(i) = A(:,i)'*B(:,i); 
end