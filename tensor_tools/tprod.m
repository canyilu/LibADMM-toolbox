function C = tprod(A,B)

% Tensor-tensor product of two 3 way tensors: C = A*B
% A - n1*n2*n3 tensor
% B - n2*l*n3  tensor
% C - n1*l*n3  tensor
%
% version 2.0 - 09/10/2017
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

[n1,n2,n3] = size(A);
[m1,m2,m3] = size(B);

if n2 ~= m1 || n3 ~= m3 
    error('Inner tensor dimensions must agree.');
end

A = fft(A,[],3);
B = fft(B,[],3);
C = zeros(n1,m2,n3);

% first frontal slice
C(:,:,1) = A(:,:,1)*B(:,:,1);
% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    C(:,:,i) = A(:,:,i)*B(:,:,i);
    C(:,:,n3+2-i) = conj(C(:,:,i));
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    C(:,:,i) = A(:,:,i)*B(:,:,i);
end
C = ifft(C,[],3);