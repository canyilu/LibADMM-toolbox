function [x,e,obj,err,iter] = tracelassoR(A,b,lambda,opts)

% Solve the trace Lasso regularized minimization problem by M-ADMM
%
% min_{x,e} loss(e)+lambda*||A*Diag(x)||_*, s.t. Ax+e=b
% loss(e) = ||e||_1 or 0.5*||e||_2^2
% ---------------------------------------------
% Input:
%       A       -    d*n matrix
%       b       -    d*1 vector
%       opts    -    Structure value in Matlab. The fields are
%           opts.loss       -   'l1' (default): loss(e) = ||e||_1 
%                               'l2': loss(e) = 0.5*||e||_2^2
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       x       -    n*1 vector
%       e       -    d*1 vector
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
loss = 'l1';

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

[d,n] = size(A);
x = zeros(n,1);
Z = zeros(d,n);
e = zeros(d,1);
Y1 = e;
Y2 = Z;

Atb = A'*b;
AtA = A'*A;
invAtA = (AtA+diag(diag(AtA)))\eye(n);

iter = 0;
for iter = 1 : max_iter
    xk = x;
    ek = e;
    Zk = Z;    
    % first super block {Z,e}
    [Z,nuclearnorm] = prox_nuclear(A*diag(x)-Y2/mu,lambda/mu);
    if strcmp(loss,'l1')
        e = prox_l1(b-A*x-Y1/mu,1/mu);
    elseif strcmp(loss,'l2')
        e = mu*(b-A*x-Y1/mu)/(1+mu);
    else
        error('not supported loss function');
    end    
    % second super block {x}
    x = invAtA*(-A'*(Y1/mu+e)+Atb+diagAtB(A,Y2/mu+Z));
    dY1 = A*x+e-b;
    dY2 = Z-A*diag(x);
    chgx = max(abs(xk-x));
    chge = max(abs(ek-e));
    chgZ = max(max(abs(Zk-Z)));
    chg = max([chgx chge chgZ max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            obj = comp_loss(e,loss)+lambda*nuclearnorm;    
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
obj = comp_loss(e,loss)+lambda*nuclearnorm;
err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);

function v = diagAtB(A,B)
% A, B - d*n matrices
% v = diag(A'*B), n*1 vector

n = size(A,2);
v = zeros(n,1);
for i = 1 : n
   v(i) = A(:,i)'*B(:,i); 
end
