function [x,e]= cappedsimplexprojection_matlab(y0,k)

% This subroutine solves the capped simplex projection problem
% min 0.5||x-y0||, s.t. 0<=x<=1, sum x_i = k;
% Reference: Weiran Wang, Canyi Lu, Projection onto the Capped Simplex, arXiv:1503.01002.
 

n=length(y0);
x=zeros(n,1);

if (k<0) || (k>n)
  error('the sum constraint is infeasible!\n');
end

if k==0;
  e=0.5*sum((x-y0).^2);
  return;
end

if k==n
  x=ones(n,1);
  e=0.5*sum((x-y0).^2);
  return;
end
[y,idx]=sort(y0,'ascend');

% Test the possiblity of a==b are integers.
if k==round(k)
  b=n-k;
  if y(b+1)-y(b)>=1
    x(idx(b+1:end))=1;
    e=0.5*sum((x-y0).^2);
    return;
  end
end

% Assume a=0.
s=cumsum(y);
y=[y;inf];
for b=1:n
  % Hypothesized gamma.
  gamma = (k+b-n-s(b)) / b;
  if ((y(1)+gamma)>0) && ((y(b)+gamma)<1) && ((y(b+1)+gamma)>=1)
    xtmp=[y(1:b)+gamma; ones(n-b,1)];
    x(idx)=xtmp;
    e=0.5*sum((x-y0).^2);
    return;
  end
end

% Now a>=1;
for a=1:n
  for b=a+1:n
    % Hypothesized gamma.
    gamma = (k+b-n+s(a)-s(b))/(b-a);
    if ((y(a)+gamma)<=0) && ((y(a+1)+gamma)>0) && ((y(b)+gamma)<1) && ((y(b+1)+gamma)>=1)
      xtmp=[zeros(a,1); y(a+1:b)+gamma; ones(n-b,1)];
      x(idx)=xtmp;
      e=0.5*sum((x-y0).^2);
      return;
    end
  end
end


