function [P,obj,err,iter] = sparsesc(L,lambda,k,opts)

% Solve the Sparse Spectral Clustering problem
%
% min_P <P,L>+lambda*||P||_1, s.t. 0\preceq P \preceq I, Tr(P)=k
%
% Reference: Canyi Lu, Shuicheng Yan, Zhouchen Lin, Convex Sparse Spectral
% Clustering: Single-view to Multi-view, TIP, 2016
%
% ---------------------------------------------
% Input:
%       L       -    n*n normalized Laplacian matrix matrix
%       k       -    integer
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       P       -    n*n matrix
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


n = size(L,1);
P = zeros(n);
Q = P;
Y = P;

iter = 0;
for iter = 1 : max_iter
    Pk = P;
    Qk = Q;
    % update P
    P = prox_l1(Q-(Y+L)/mu,lambda/mu);
    % update Q
    temp = P+Y/mu;
    temp = (temp+temp')/2;
    Q = project_fantope(temp,k);
    
    dY = P-Q;
    chgP = max(max(abs(Pk-P)));
    chgQ = max(max(abs(Qk-Q)));
    chg = max([chgP chgQ max(abs(dY(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            obj = trace(P'*L)+lambda*norm(Q(:),1);
            err = norm(dY,'fro');
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
obj = trace(P'*L)+lambda*norm(Q(:),1);
err = norm(dY,'fro');