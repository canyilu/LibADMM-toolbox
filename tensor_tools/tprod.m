function C = tprod(A,B)

% Tensor-Tensor product of two 3-way tensor: C = A*B
% A - n1*n2*n3 tensor
% B - n2*l*n3  tensor
% C - n1*l*n3  tensor
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

[n1,~,n3] = size(A);
l = size(B,2);
A = fft(A,[],3);
B = fft(B,[],3);
C = zeros(n1,l,n3);
for i = 1 : n3
    C(:,:,i) = A(:,:,i)*B(:,:,i);
end
C = ifft(C,[],3);




