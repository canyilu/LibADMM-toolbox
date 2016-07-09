addpath(genpath(cd))
clear


pic_name = [ './image/testimg.jpg'];
X = double(imread(pic_name));
  
X = X/255;
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
Xn = X;
rhos = 0.3
ind = find(rand(n1*n2*n3,1)<rhos);
Xn(ind) = rand(length(ind),1);

opts.mu = 1e-4;
opts.tol = 1e-5;
opts.rho = 1.2;
opts.max_iter = 500;
opts.DEBUG = 1;

%% RPCA
% lambda = 1/(sqrt(max(n1,n2)));
% Xhat = zeros(n1,n2,n3);
% Shat = Xhat;
% 
% tol = 1e-6;
% maxIter = 500;
% mu = 1e-3;
% for j = 1 : 3
%     Xni = Xn(:,:,j);
%     [Xhat(:,:,j),S,obj,err,iter] = rpca(Xni,lambda,opts);
% end
% Xhat = max(Xhat,0);
% Xhat = min(Xhat,maxP);
% Shat = max(Shat,0);
% Shat = min(Shat,maxP);
% Lr_RPCA = norm(X(:)-Xhat(:))/norm(X(:));
% psnr_RPCA = PSNR(X,Xhat,maxP)
% 
% 
% figure(2)
% subplot(1,3,1)
% imshow(X/max(X(:)))
% subplot(1,3,2)
% imshow(Xn/max(Xn(:)))
% subplot(1,3,3)
% imshow(Xhat/max(Xhat(:)))

%% Tensor RRPCA based on SNN
% alpha = [15 15 1.5];
%  
% [Xhat,E,err,iter] = trpca_snn(Xn,alpha,opts);
% 
% err
% iter
%  
% Xhat = max(Xhat,0);
% Xhat = min(Xhat,maxP);
% psnr = PSNR(X,Xhat,maxP)
% 
% figure(1)
% subplot(1,3,1)
% imshow(X/max(X(:)))
% subplot(1,3,2)
% imshow(Xn/max(Xn(:)))
% subplot(1,3,3)
% imshow(Xhat/max(Xhat(:)))
% 
% pause 

%% Tensor RRPCA based on TNN
[n1,n2,n3] = size(Xn);
lambda = 1/sqrt(max(n1,n2)*n3);
[Xhat,E,err,iter] = trpca_tnn(Xn,lambda,opts);

err
iter
 
Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
psnr = PSNR(X,Xhat,maxP)

figure(1)
subplot(1,3,1)
imshow(X/max(X(:)))
subplot(1,3,2)
imshow(Xn/max(Xn(:)))
subplot(1,3,3)
imshow(Xhat/max(Xhat(:)))



