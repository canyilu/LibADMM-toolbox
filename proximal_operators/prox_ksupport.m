function B = prox_ksupport(v,k,lambda)

% The proximal operator of the k support norm of a vector
%
% min_x 0.5*lambda*||x||_{ksp}^2+0.5*||x-v||_2^2
%
% version 1.0 - 27/06/2016
%
% Written by Hanjiang Lai
%
% Reference: 
% Lai H, Pan Y, Lu C, et al. Efficient k-support matrix pursuit, ECCV, 2014: 617-631.
% 

L = 1/lambda;
d = length(v);
if k >= d
    B = L*v/(1+L);
    return;
elseif k <= 1
    k = 1;
end

[z, ind] = sort(abs(v), 'descend');
z = z*L;
ar = cumsum(z);
z(d+1) = -inf;

diff = 0;
err = inf;
found = false;
for r=k-1:-1:0
    [l,T] = bsearch(z,ar,k-r,d,diff,k,r,L);
    if ( ((L+1)*T >= (l-k+(L+1)*r+L+1)*z(k-r)) && ...
            (((k-r-1 == 0) || (L+1)*T < (l-k+(L+1)*r+L+1)*z(k-r-1)) ) )
        found = true;
        break;
    end
    diff = diff + z(k-r);
    if k-r-1 == 0
        err_tmp = max(0,(l-k+(L+1)*r+L+1)*z(k-r) - (L+1)*T);
    else
        err_tmp = max(0,(l-k+(L+1)*r+L+1)*z(k-r) -(L+1)*T) + max(0, - (l-k+(L+1)*r+L+1)*z(k-r-1) + (L+1)*T);
    end
    if err > err_tmp
        err_r = r; err_l = l; err_T = T; err = err_tmp;
    end
end


if found == false
    r = err_r; l = err_l; T = err_T;
end

%  fprintf('r = %d, l = %d \n',r,l);

p(1:k-r-1) = z(1:k-r-1)/(L+1);
p(k-r:l) = T / (l-k+(L+1)*r+L+1);
p(l+1:d) = z(l+1:d);
p = p';

% [dummy, rev]=sort(ind,'ascend');
rev(ind) = 1:d;
p = sign(v) .* p(rev);
B = v - 1/L*p;
end

function [l,T] = bsearch(z,array,low,high,diff,k,r,L)
if z(low) == 0
    l = low;
    T = 0;
    return;
end
%z(mid) * tmp - (array(mid) - diff) > 0
%z(mid+1) * tmp - (array(mid+1) - diff) <= 0
while( low < high )
    mid = floor( (low + high)/2 ) + 1;
    tmp = (mid - k + r + 1 + L*(r+1));
    if z(mid) * tmp - (array(mid) - diff) > 0
        low = mid;
    else
        high = mid - 1;
    end
end
l = low;
T = array(low) - diff;
end


