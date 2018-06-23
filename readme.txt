LibADMM: A Library of ADMM for Sparse and Low-rank Optimization


This package solves several sparse and low-rank optimization problems by M-ADMM proposed in our work
C. Lu, J. Feng, S. Yan, Z. Lin. A Unified Alternating Direction Method of Multipliers by Majorization Minimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 527-541, 2018
    

The folder "LibADMM" contains three subfolders:

1. algorithms: the main solvers.
2. proximal_operators: the proximal operators of several functions used in the subproblems of M-ADMM.
3. tensor_tools: some basic tools for tensors.

Besides the subfolders, we also three functions, "test_sparse_models.m", "test_low_rank_matrix_models.m", and "test_low_rank_tensor_models.m" which provide the examples for all the solvers implemented in this package.

You are also suggested to read the manual at https://canyilu.github.io/publications/2016-software-LibADMM.pdf.

For any problems, please contact Canyi Lu (canyilu@gmail.com).


Version 1.0 (Jun, 2016)

Version 1.1 (Jun, 2018)
- add a new model low-rank tensor recovery from Gaussian measurements based on tensor nuclear norm and the corresponding function lrtr_Gaussian_tnn.m
- update several functions to improve the efficiency, including prox_tnn.m, tprod.m, tran.m, tubalrank.m, and nmodeproduct.m
- update the three example functions: example_sparse_models.m, example_low_rank_matrix_models.m, and example_low_rank_tensor_models.m
- remove the test on image data and some unnecessary functions
