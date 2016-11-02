function [S, V, D, Sigma2] = MySVD(A)
[m, n] = size(A);
if 2*m < n
    AAT = A*A';
    [S, Sigma2, D] = svd(AAT);
    Sigma2 = diag(Sigma2);
    V = sqrt(Sigma2);
    tol = max(size(A)) * eps(max(V));
    R = sum(V > tol);
    V = V(1:R);
    S = S(:,1:R);
    D = A'*S*diag(1./V);
    V = diag(V);
    return;
end
if m > 2*n
    [S, V, D, Sigma2] = MySVD(A');
    mid = D;
    D = S;
    S = mid;
    return;
end
[S,V,D] = svd(A);
Sigma2 = diag(V).^2;