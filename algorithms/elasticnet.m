function [X,obj,err,iter] = elasticnet(A,B,lambda,opts)

% Solve the elastic net minimization problem by ADMM
%
% min_X ||X||_1+lambda*||X||_F^2, s.t. AX=B
%
% ---------------------------------------------
% Input:
%       A       -    d*na matrix
%       B       -    d*nb matrix
%       lambda  -    >=0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    na*nb matrix
%       obj     -    objective function value
%       err     -    residual ||AX-B||_F
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


[d,na] = size(A);
[~,nb] = size(B);

X = zeros(na,nb);
Z = X;
Y1 = zeros(d,nb);
Y2 = X;

AtB = A'*B;
I = eye(na);
invAtAI = (A'*A+I)\I;

iter = 0;
for iter = 1 : max_iter
    Xk = X;
    Zk = Z;
    % update X
    X = prox_elasticnet(Z-Y2/mu,1/mu,lambda/mu);
    % update Z
    Z = invAtAI*(-(A'*Y1-Y2)/mu+AtB+X);    
    dY1 = A*Z-B;
    dY2 = X-Z;
    chgX = max(max(abs(Xk-X)));
    chgZ = max(max(abs(Zk-Z)));
    chg = max([chgX chgZ max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            obj = norm(X(:),1)+lambda*norm(X,'fro')^2;
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
obj = norm(X(:),1)+lambda*norm(X,'fro')^2;
err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);

