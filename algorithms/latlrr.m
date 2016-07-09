function [Z,L,obj,err,iter] = latlrr(X,lambda,opts)

% Solve the Latent Low-Rank Representation by M-ADMM
%
% min_{Z,L,E} ||Z||_*+||L||_*+lambda*loss(E),
% s.t., XZ+LX-X=E.
% loss(E) = ||E||_1 or 0.5*||E||_F^2 or ||E||_{2,1}
% ---------------------------------------------
% Input:
%       X       -    d*n matrix
%       lambda  -    >0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.loss       -   'l1' (default): loss(E) = ||E||_1 
%                               'l2': loss(E) = 0.5*||E||_F^2
%                               'l21': loss(E) = ||E||_{2,1}
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       Z       -    n*n matrix
%       L       -    d*d matrix
%       E       -    d*n matrix
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

eta1 = 1.02*2*norm(X,2)^2; % for Z
eta2 = eta1; % for L
eta3 = 1.02*2; % for E

[d,n] = size(X);
E = zeros(d,n);
Z = zeros(n,n);
L = zeros(d,d);
Y = E;

XtX = X'*X;
XXt = X*X';

iter = 0;
for iter = 1 : max_iter
    Lk = L;
    Ek = E;
    Zk = Z;
    % first super block {Z}
    [Z,nuclearnormZ] = prox_nuclear(Zk-(X'*(Y/mu+L*X-X-E)+XtX*Z)/eta1,1/(mu*eta1));
    % second super block {L,E}
    temp = Lk-((Y/mu+X*Z-Ek)*X'+Lk*XXt-XXt)/eta2;
    [L,nuclearnormL] = prox_nuclear(temp,1/(mu*eta2));        
    if strcmp(loss,'l1')
        E = prox_l1(Ek+(Y/mu+X*Z+Lk*X-X-Ek)/eta3,lambda/(mu*eta3));
    elseif strcmp(loss,'l21')
        E = prox_l21(Ek+(Y/mu+X*Z+Lk*X-X-Ek)/eta3,lambda/(mu*eta3));
    elseif strcmp(loss,'l2')
        E = (Y+mu*(X*Z+Lk*X-X+(eta3-1)*Ek))/(lambda+mu*eta3);
    else
        error('not supported loss function');
    end
    
    dY = X*Z+L*X-X-E;
    chgL = max(max(abs(Lk-L)));
    chgE = max(max(abs(Ek-E)));
    chgZ = max(max(abs(Zk-Z)));
    chg = max([chgL chgE chgZ max(abs(dY(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            obj = nuclearnormZ+nuclearnormL+lambda*comp_loss(E,loss);
            err = norm(dY,'fro')^2;
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);    
end
obj = nuclearnormZ+nuclearnormZ+lambda*comp_loss(E,loss);
err = norm(dY,'fro')^2;

function out = comp_loss(E,loss)

switch loss
    case 'l1'
        out = norm(E(:),1);
    case 'l21'
        out = 0;
        for i = 1 : size(E,2)
            out = out + norm(E(:,i));
        end
    case 'l2'
        out = 0.5*norm(E,'fro')^2;
end

 