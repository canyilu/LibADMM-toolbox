function [L,S,obj,err,iter] = igc(A,C,lambda,opts)

% Reference: Chen, Yudong, Sujay Sanghavi, and Huan Xu. Improved graph clustering.
% IEEE Transactions on Information Theory 60.10 (2014): 6440-6455.
%
% min_{L,S} ||L||_*+lambda*||C \cdot S||_1, s.t. A=L+S, 0<=L<=1.
%
% ---------------------------------------------
% Input:
%       A       -    d*n matrix
%       C       -    d*n matrix
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
%       S       -    d*n matrix
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

C = abs(C);
[d,n] = size(A);

L = zeros(d,n);
S = L;
Z = L;
Y1 = L;
Y2 = L;

iter = 0;
for iter = 1 : max_iter
    Lk = L;
    Sk = S;
    Zk = Z;
    % first super block {L,S}
    [L,nuclearnormL] = prox_nuclear(Z-Y2/mu,1/mu);
    S = prox_l1(-Z+A-Y1/mu,C*(lambda/mu));
    
    % second super block {Z}
    Z = project_box((-S+A+L+(Y2-Y1)/mu)/2,0,1);
  
    dY1 = Z+S-A;
    dY2 = L-Z;
    chgL = max(max(abs(Lk-L)));
    chgS = max(max(abs(Sk-S)));
    chgZ = max(max(abs(Zk-Z)));
    chg = max([chgL chgS chgZ max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            obj = nuclearnormL+lambda*sum(sum(C.*abs(S)));
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
obj = nuclearnormL+lambda*sum(sum(C.*abs(S)));
err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2);

