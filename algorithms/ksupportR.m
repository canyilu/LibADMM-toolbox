function [X,E,err,iter] = ksupportR(A,B,lambda,k,opts)

% Solve the l1 norm regularized minimization problem by M-ADMM
%
% min_{X,E} loss(E)+0.5*||vec(X)||_ksp^2, s.t. AX+E=B
% loss(E) = ||E||_1 or 0.5*||E||_F^2
% ---------------------------------------------
% Input:
%       A       -    d*na matrix
%       B       -    d*nb matrix
%       lambda  -    >=0, parameter
%       k       -    >0, integer, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.loss       -   'l1' (default): loss(E) = ||E||_1 
%                               'l2': loss(E) = 0.5*||E||_F^2
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    na*nb matrix
%       E       -    d*nb matrix
%       obj     -    objective function value
%       err     -    residual 
%       iter    -    number of iterations
%
% version 1.0 - 27/06/2016
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


[d,na] = size(A);
[~,nb] = size(B);

X = zeros(na,nb);
E = zeros(d,nb);
Z = X;
Y1 = E;
Y2 = X;

AtB = A'*B;
I = eye(na);
invAtAI = (A'*A+I)\I;

iter = 0;
for iter = 1 : max_iter
    Xk = X;
    Ek = E;
    Zk = Z;
    % first super block {X,E}
    temp = Z-Y2/mu;
    temp = prox_ksupport(temp(:),k,lambda/mu);
    X = reshape(temp,na,nb);
    if strcmp(loss,'l1')
        E = prox_l1(B-A*Z-Y1/mu,1/mu);
    elseif strcmp(loss,'l2')
        E = mu*(B-A*Z-Y1/mu)/(1+mu);
    else
        error('not supported loss function');
    end
    % second  super block {Z}
    Z = invAtAI*(-A'*(Y1/mu+E)+AtB+Y2/mu+X);    
    dY1 = A*Z+E-B;
    dY2 = X-Z;
    chgX = max(max(abs(Xk-X)));
    chgE = max(max(abs(Ek-E)));
    chgZ = max(max(abs(Zk-Z)));
    chg = max([chgX chgE chgZ max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y1 = Y1 + mu*dY1;
    Y2 = Y2 + mu*dY2;
    mu = min(rho*mu,max_mu);    
end
err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);

