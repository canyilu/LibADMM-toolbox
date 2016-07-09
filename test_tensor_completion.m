addpath(genpath(cd))
clear

pic_name = [ './image/testimg.jpg'];
I = double(imread(pic_name));
X = I/255;
    
[n1,n2,n3] = size(X);

opts.mu = 1e-3;
opts.tol = 1e-6;
opts.rho = 1.2;
opts.max_iter = 500;
opts.DEBUG = 1;


p = 0.6;
maxP = max(abs(X(:)));
omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);


%% %% test lrtc_snn and lrtc_tnn

alpha = [1, 1, 1e-3];
alpha = alpha / sum(alpha);
[Xhat,err,iter] = lrtc_snn(M,omega,alpha,opts);
% [Xhat,err,iter] = lrtc_tnn(M,omega,opts);

err
iter
 
Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
RSE = norm(X(:)-Xhat(:))/norm(X(:))
psnr = PSNR(X,Xhat,maxP)

figure(1)
subplot(1,3,1)
imshow(X/maxP)
subplot(1,3,2)
imshow(M/maxP)
subplot(1,3,3)
imshow(Xhat/maxP)

pause

%% test lrtcR_snn
E = randn(n1,n2,n3)/100;
M = M+E; 

alpha = [1, 1, 0.001]*10;
% alpha = alpha / sum(alpha);

[Xhat,err,iter] = lrtcR_snn(M,omega,alpha,opts);
err
iter
 

Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
RSE = norm(X(:)-Xhat(:))/norm(X(:))
psnr = PSNR(X,Xhat,maxP)

figure(1)
subplot(1,3,1)
imshow(X/maxP)
subplot(1,3,2)
imshow(M/maxP)
subplot(1,3,3)
imshow(Xhat/maxP)

pause
%% test lrtcR_tnn

E = randn(n1,n2,n3)/100;
M = M+E; 

lambda = 0.1;
[Xhat,E,obj,err,iter] = lrtcR_tnn(M,omega,lambda,opts);
err
iter
 

Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
RSE = norm(X(:)-Xhat(:))/norm(X(:))
psnr = PSNR(X,Xhat,maxP)

figure(1)
subplot(1,3,1)
imshow(X/maxP)
subplot(1,3,2)
imshow(M/maxP)
subplot(1,3,3)
imshow(Xhat/maxP)
