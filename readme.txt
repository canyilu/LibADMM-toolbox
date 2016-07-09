LibADMM package Version 1.0 (Jun, 2016)

This package solves many popular compressive sensing problems (see problem_list.pdf) by M-ADMM proposed in

Canyi Lu, Jiashi Feng, Shuicheng Yan, Zhouchen Lin. A Unified Alternating Direction Method of Multipliers by Majorization Minimization. In submission.     

The folder "LibADMM" contains four subfolders:

1. algorithsm: the main solvers.
2. proximal_operator: the proximal operators of several functions used in the subproblems of M-ADMM.
3. tensor_tools: some basic tools for tensors.
4. image: image used for testing solvers.

Besides the subfolders, we also provide four functions, "test_sparse_models.m", "test_lowrankmatrix_models.m", "test_tensor_completion.m" and "test_tensor_rpca.m" which provide the examples for all the solvers implemented in this package.

For any problems, please contact Canyi Lu (canyilu@gmail.com).