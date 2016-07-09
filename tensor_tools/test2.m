

n1 = 3;
n2 = 3;
n3 = 3;
l = 3;
A = rand(n1,n2,n3);
B = rand(n2,l,n3);

C = tprod(A,B);
C2 = tprod2(A,B);

norm(C(:)-C2(:))

At = tran(A)
At2 = tran2(A)
norm(At(:)-At2(:))

[U,S,V,trank] = tsvd(A);
trank
Ar = tprod(tprod(U,S),tran2(V));

norm(A(:)-Ar(:))

% U

UtU = tprod(tran(U),U)

dif = UtU - tI(size(UtU,1),size(UtU,3));
norm(dif(:))


C = tprod(A,tran(B));
Ab = fft(A,[],3);
Ab = tbdiag(Ab);
Bb = fft(tran(B),[],3);
Bb = tbdiag(Bb);
Cb = Ab*Bb;
C2 = itbdiag(Cb,n1,n2,n3);
C2 = ifft(C2,[],3);
norm(C(:)-C2(:))

Bt = rand(n1,n2,n3)-i*rand(n1,n2,n3);
Bt = ifft(Bt,[],3)



[U,S,V,trank,tnn] = tsvd(A);

s = S(:,:,1);
tnn - sum(s(:))


