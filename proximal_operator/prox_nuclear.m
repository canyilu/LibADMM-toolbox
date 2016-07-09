function [X,nuclearnorm] = prox_nuclear(B,lambda)

% The proximal operator of the nuclear norm of a matrix
% 
% min_X lambda*||X||_*+0.5*||X-B||_F^2
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

[U,W,V] = mySVD(B);
W = diag(W);
svp = length(find(W>lambda));
if svp>=1
    W = W(1:svp)-lambda;
    X = U(:,1:svp)*diag(W)*V(:,1:svp)';
    nuclearnorm = sum(W);
else
    X = zeros(size(B));
    nuclearnorm = 0;
end