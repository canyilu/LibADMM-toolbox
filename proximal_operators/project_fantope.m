function X = project_fantope(Q,k)

% Project a point onto the Fantope
% Q - a symmetric matrix
%
% min_X ||X-Q||_F, s.t. 0\succeq X \succeq I, Tr(X)=k.
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

[U,D] = eig(Q);
Dr = cappedsimplexprojection(diag(D),k);
% Dr = cappedsimplexprojection_matlab(diag(D),k);
X = U*diag(Dr)*U';