function x = prox_gl1(b,G,lambda)

% The proximal operator of the group l1 norm
% 
% min_x lambda*\sum_{g in G} ||x_g||_2+0.5*||x-b||_2^2
% ---------------------------------------------
% Input:
%       b       -    d*1 vector
%       G       -    a cell indicates a partition of 1:d
%
% Output:
%       x       -    d*1 vector
% 
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

x = zeros(size(b));
for i = 1 : length(G)
    nxg = norm(b(G{i}));
    if nxg > lambda  
        x(G{i}) = b(G{i})*(1-lambda/nxg);
    end
end