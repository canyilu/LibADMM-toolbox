function [x,e,obj,err,iter] = fusedl1R(A,b,lambda1,lambda2,opts)

% Solve the fused Lasso regularized minimization problem by ADMM
%
% min_{x,e} loss(e) + lambda1*||x||_1 + lambda2*\sum_{i=2}^p |x_i-x_{i-1}|,
% loss(e) = ||e||_1 or 0.5*||e||_2^2
%
% ---------------------------------------------
% Input:
%       A       -    d*n matrix
%       b       -    d*1 vector
%       lambda1 -    >=0, parameter
%       lambda2 -    >=0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.loss       -   'l1' (default): loss(e) = ||e||_1 
%                               'l2': loss(E) = 0.5*||e||_2^2
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
% version 1.0 - 20/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;
DEBUG = 0;
loss = 'l1'; % default

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
e = zeros(d,1);
z = x;
Y1 = e;
Y2 = x;

Atb = A'*b;
I = eye(n);
invAtAI = (A'*A+I)\I;


% parameters for "flsa" (from SLEP package)
tol2 = 1e-10;      % the duality gap for termination
max_step = 50;     % the maximal number of iterations
x0 = zeros(n-1,1); % the starting point

iter = 0;
for iter = 1 : max_iter
    xk = x;
    ek = e;
    zk = z;
    % first super block {x,e}
    % flsa solves min_x 1/2||x-v||_2^2+lambda1*||x||_1+lambda2*\sum_{i=2}^p |x_i-x_{i-1}|,
    x = flsa(z-Y2/mu,x0,lambda1/mu,lambda2/mu,n,max_step,tol2,1,6);
    if strcmp(loss,'l1')
        e = prox_l1(b-A*z-Y1/mu,1/mu);
    elseif strcmp(loss,'l2')
        e = mu*(b-A*z-Y1/mu)/(1+mu);
    else
        error('not supported loss function');
    end
    % second  super block {Z}
    z = invAtAI*(-A'*(Y1/mu+e)+Atb+Y2/mu+x);    
    dY1 = A*z+e-b;
    dY2 = x-z;
    chgx = max(abs(xk-x));
    chge = max(abs(ek-e));
    chgz = max(abs(zk-z));
    chg = max([chgx chge chgz max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0            
            obj = comp_loss(e,loss)+comp_fusedl1(x,lambda1,lambda2);
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
obj = comp_loss(e,loss)+comp_fusedl1(x,lambda1,lambda2);
err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);


function f = comp_fusedl1(x,lambda1,lambda2)
% compute f = lambda1*||x||_1 + lambda2*\sum_{i=2}^p |x_i-x_{i-1}|.
% x - p*1 vector
f = 0;
p = length(x);
for i = 2 : p
   f = f+abs(x(i)-x(i-1)); 
end
f = lambda1*norm(x,1)+lambda2*f;


