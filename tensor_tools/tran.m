function Xt = tran(X)

% conjugate transpose of a 3 way tensor 
% X  - n1*n2*n3 tensor
% Xt - n2*n1*n3  tensor
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

[n1,n2,n3] = size(X);
Xt = zeros(n2,n1,n3);
Xt(:,:,1) = X(:,:,1)';
if n3 > 1
    for i = 2 : n3
        Xt(:,:,i) = X(:,:,n3-i+2)';
    end
end
