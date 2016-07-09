function [X,E,obj,err,iter] = lrsr(A,B,lambda1,lambda2,opts)

% Solve the Low-Rank and Sparse Representation (LRSR) minimization problem by M-ADMM
%
% min_{X,E} ||X||_*+lambda1*||X||_1+lambda2*loss(E), s.t. A=BX+E
% loss(E) = ||E||_1 or 0.5*||E||_F^2 or ||E||_{2,1}
% ---------------------------------------------
% Input:
%       A       -    d*na matrix
%       B       -    d*nb matrix
%       lambda1 -    >0, parameter
%       lambda2 -    >0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.loss       -   'l1': loss(E) = ||E||_1 
%                               'l2': loss(E) = 0.5*||E||_F^2
%                               'l21' (default): loss(E) = ||E||_{2,1}
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    nb*na matrix
%       E       -    d*na matrix
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
loss = 'l21';

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

X = zeros(nb,na);
E = zeros(d,na);
Z = X;
J = X;

Y1 = E;
Y2 = X;
Y3 = X;
BtB = B'*B;
BtA = B'*A;
I = eye(nb);
invBtBI = (BtB+2*I)\I;

iter = 0;
for iter = 1 : max_iter
    Xk = X;
    Zk = Z;
    Ek = E;
    Jk = J;
    % first super block {Z,J,E}
    [Z,nuclearnormZ] = prox_nuclear(X+Y2/mu,1/mu);
    J = prox_l1(X+Y3/mu,lambda1/mu);
    if strcmp(loss,'l1')
        E = prox_l1(A-B*X+Y1/mu,lambda2/mu);
    elseif strcmp(loss,'l21')
        E = prox_l21(A-B*X+Y1/mu,lambda2/mu);
    elseif strcmp(loss,'l2')
        E = mu*(A-B*X+Y1/mu)/(lambda2+mu);
    else
        error('not supported loss function');
    end
    % second  super block {X}
    X = invBtBI*(B'*(Y1/mu-E)+BtA-(Y2+Y3)/mu+Z+J);
  
    dY1 = A-B*X-E;
    dY2 = X-Z;
    dY3 = X-J;
    chgX = max(max(abs(Xk-X)));
    chgE = max(max(abs(Ek-E)));
    chgZ = max(max(abs(Zk-Z)));
    chgJ = max(max(abs(Jk-J)));
    chg = max([chgX chgE chgZ chgJ max(abs(dY1(:))) max(abs(dY2(:))) max(abs(dY3(:)))]);
    if DEBUG        
        if iter == 1 || mod(iter, 10) == 0
            obj = nuclearnormZ+lambda1*norm(J(:),1)+lambda2*comp_loss(E,loss);
            err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2+norm(dY3,'fro')^2);
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
obj = nuclearnormZ+lambda1*norm(J(:),1)+lambda2*comp_loss(E,loss);
err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2+norm(dY3,'fro')^2);


function out = comp_loss(E,normtype)

switch normtype
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
