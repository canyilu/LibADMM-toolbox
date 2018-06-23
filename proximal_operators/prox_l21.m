function X = prox_l21(B,lambda)

% The proximal operator of the l21 norm of a matrix
% l21 norm is the sum of the l2 norm of all columns of a matrix 
%
% min_X lambda*||X||_{2,1}+0.5*||X-B||_2^2
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
%

X = zeros(size(B));
for i = 1 : size(X,2)
    nxi = norm(B(:,i));
    if nxi > lambda  
        X(:,i) = (1-lambda/nxi)*B(:,i);
    end
end