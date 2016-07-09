function [U,S,V,trank,tnn] = tsvd(Y)

% T-SVD of a 3-way tensor, Y=U*S*V^*
%
% Y  - n1*n2*n3 tensor
% U  - n1*r*n3  tensor
% S  - r*r*n3 tensor
% V  - n2*r*n3 tensor
% trank - tubal rank of Y
% tnn   - tensor nuclear norm
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

[n1,n2,n3] = size(Y);
n12 = min(n1,n2);
Y = fft(Y,[],3);
U = zeros(n1,n12,n3);
V = zeros(n2,n12,n3);
S = zeros(n12,n12,n3);
trank = 0;
for i = 1 : n3
    [U(:,:,i),s,V(:,:,i)] = svd(Y(:,:,i),'econ');
    s = diag(s);
    S(:,:,i) = diag(s);
    tranki = length(find(s~=0));
    trank = max(tranki,trank);
end
U = U(:,1:trank,:);
V = V(:,1:trank,:);
S = S(1:trank,1:trank,:);

U = ifft(U,[],3);
S = ifft(S,[],3);
V = ifft(V,[],3);

tnn = sum(sum(S(:,:,1)));

