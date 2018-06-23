function trank = tubalrank(X,tol)

% The tensor tubal rank of a 3 way tensor
%
% X     -    n1*n2*n3 tensor
% trank -    tensor tubal rank of X
%
% version 2.0 - 14/06/2018
%
% Written by Canyi Lu (canyilu@gmail.com)
%
%
% References: 
% Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
% June, 2018. https://github.com/canyilu/tproduct.
%
% Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
% Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
% Norm, arXiv preprint arXiv:1804.03728, 2018
%

X = fft(X,[],3);
[n1,n2,n3] = size(X);
s = zeros(min(n1,n2),1);

% i=1
s = s + svd(X(:,:,1),'econ');
% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    s = s + svd(X(:,:,i),'econ')*2;
end
% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    s = s + svd(X(:,:,i),'econ');
end
s = s/n3;

if nargin==1
   tol = max(n1,n2) * eps(max(s));
end
trank = sum(s > tol);
