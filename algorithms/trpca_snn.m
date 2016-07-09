function [L,E,err,iter] = trpca_snn(X,alpha,opts)

% Solve the Tensor Robust Principal Component Analysis (TRPCA) based on Sum of Nuclear Norm (SNN) problem by M-ADMM
%
% min_{L,E} \sum_i \alpha_i*||L_{i(i)}||_* + ||E||_1,
% s.t. X = L + E.
%
% ---------------------------------------------
% Input:
%       X       -    d1*d2*...dk tensor
%       alpha   -    k*1 vector, parameters
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       L       -    d1*d2*...*dk tensor
%       E       -    d1*d2*...*dk tensor
%       err     -    residual
%       iter    -    number of iterations
%
% version 1.0 - 24/06/2016
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

dim = size(X);
k = length(dim);

E = zeros(dim);
Y = cell(k,1);
L = Y;
for i = 1 : k
    Y{i} = E;
    L{i} = E;
end

iter = 0;
for iter = 1 : max_iter
    Lk = L;
    Ek = E;
    % first super block {L_i}
    sumtemp = zeros(dim);
    for i = 1 : k
        L{i} = Fold(prox_nuclear(Unfold(X-E-Y{i}/mu,dim,i), alpha(i)/mu),dim,i);
        sumtemp = sumtemp + L{i} + Y{i}/mu;
    end
    % second super block {E}
    E = prox_l1(X-sumtemp/k,1/(mu*k));
    
    chg = max(abs(Ek(:)-E(:)));
    err = 0;
    for i = 1 : k
        dY = L{i}+E-X;
        err = err+norm(dY(:))^2;
        Y{i} = Y{i}+mu*dY;
        chg = max([chg, max(abs(dY(:))), max(abs(Lk{i}(:)-L{i}(:)))]);
    end
    err = sqrt(err);

    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', err=' num2str(err)]); 
        end
    end
    if chg < tol
        break;
    end 
    mu = min(rho*mu,max_mu);    
end
L = L{1};
