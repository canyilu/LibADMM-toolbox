function I = tI(n,n3)

% 3 way tensor identity tensor
% I  - n*n*n3 tensor
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

I = zeros(n,n,n3);
I(:,:,1) = eye(n);

