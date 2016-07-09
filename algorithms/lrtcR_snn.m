function [X,err,iter] = lrtcR_snn(M,omega,alpha,opts)

% Solve the Noisy Low-Rank Tensor Completion (LRTC) based on Sum of Nuclear Norm (SNN) problem by M-ADMM
%
% min_{X,E} \sum_i \alpha_i*||X_{i(i)}||_* + loss(E),
% s.t. P_Omega(X) + E = M.
% loss(E) = ||E||_1 or 0.5*||E||_F^2
%
% ---------------------------------------------
% Input:
%       M       -    d1*d2*...dk tensor
%       omega   -    index of the observed entries
%       alpha   -    k*1 vector, parameters
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
%       X       -    d1*d2*...*dk tensor
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

dim = size(M);
k = length(dim);

omegac = setdiff(1:prod(dim),omega);

X = zeros(dim);
Y = cell(k,1);
Z = Y;
E = X;
Y2 = E;
for i = 1 : k
    Y{i} = X;
    Z{i} = X;
end

iter = 0;
for iter = 1 : max_iter
    Xk = X;
    Ek = E;
    Zk = Z;
    % first super block {Z_i,E}
    sumtemp = zeros(dim);
    for i = 1 : k
        Z{i} = Fold(prox_nuclear(Unfold(X+Y{i}/mu,dim,i), alpha(i)/mu),dim,i);
        sumtemp = sumtemp + Z{i} - Y{i}/mu;
    end    
    if strcmp(loss,'l1')
        E = prox_l1(-X+M-Y2/mu,1/mu);
    elseif strcmp(loss,'l2')
        E = (-X+M-Y2/mu)*(mu/(1+mu));
    else
        error('not supported loss function');
    end
    % second super block {X}
    X(omega) = (sumtemp(omega)-Y2(omega)/mu-E(omega)+M(omega))/(k+1);
    X(omegac) = sumtemp(omegac)/k;
    
    chg = max([max(abs(Xk(:)-X(:))), max(abs(Ek(:)-E(:))) ]);
    err = 0;
    for i = 1 : k
        dY = X-Z{i};
        err = err+norm(dY(:))^2;
        Y{i} = Y{i}+mu*dY;
        chg = max([chg,max(abs(dY(:))), max(abs((Zk{i}(:)-Z{i}(:))))]);
    end
    dY = E-M;    
    dY(omega) = dY(omega)+X(omega);
    chg = max(chg,max(abs(dY(:))));
    Y2 = Y2 + mu*dY;
    err = sqrt(err+norm(dY(:))^2);

    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', err=' num2str(chg)]); 
        end
    end
    if chg < tol
        break;
    end 
    mu = min(rho*mu,max_mu);    
end
 