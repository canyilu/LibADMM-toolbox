function nnorm = nuclearnorm( X )

nnorm = sum(svd(X,'econ'));
