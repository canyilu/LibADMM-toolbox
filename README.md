## LibADMM

### Introduction

This toolbox solves many sparse, low-rank matrix and low-rank tensor optimization problems by using M-ADMM developed in our paper <a class="footnote-reference" href="#id2" id="id1">[1]</a>. 

### List of Problems

The table below gives the list of problems solved in our toolbox. See more details in the manual at <a href="../publications/2016-software-LibADMM.pdf" class="textlink" target="_blank">https://canyilu.github.io/publications/2016-software-LibADMM.pdf</a>. 

<p align="center"> 
<img src="https://github.com/canyilu/LibADMM/blob/master/tab_problemlist.JPG">
</p>

### Citing

<p>In citing this toolbox in your papers, please use the following references:</p>

<div class="highlight-none"><div class="highlight"><pre>
C. Lu, J. Feng, S. Yan, Z. Lin. A Unified Alternating Direction Method of Multipliers by Majorization 
Minimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 527-541, 2018
C. Lu. A Library of ADMM for Sparse and Low-rank Optimization. National University of Singapore, June 2016.
https://github.com/canyilu/LibADMM.
</pre></div>
  

<p>The corresponding BiBTeX citation are given below:</p>
<div class="highlight-none"><div class="highlight"><pre>
@manual{lu2016libadmm,
author       = {Lu, Canyi},
title        = {A Library of {ADMM} for Sparse and Low-rank Optimization},
organization = {National University of Singapore},
month        = {June},
year         = {2016},
note         = {\url{https://github.com/canyilu/LibADMM}}
}
@article{lu2018unified,
author       = {Lu, Canyi and Feng, Jiashi and Yan, Shuicheng and Lin, Zhouchen},
title        = {A Unified Alternating Direction Method of Multipliers by Majorization Minimization},
journal      = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
publisher    = {IEEE},
year         = {2018},
volume       = {40},
number       = {3},
pages        = {527—-541},
}</pre></div>

### Version History
- Version 1.0 was released on June, 2016.
- Version 1.1 was released on June, 2018. Some key differences are below:
  + Add a new model about low-rank tensor recovery from Gaussian measurements based on tensor nuclear norm and the corresponding function lrtr_Gaussian_tnn.m
  + Update several functions to improve the efficiency, including prox_tnn.m, tprod.m, tran.m, tubalrank.m, and nmodeproduct.m
  + Update the three example functions: example_sparse_models.m, example_low_rank_matrix_models.m, and example_low_rank_tensor_models.m
  + Remove the test on image data and some unnecessary functions

### References
<table class="docutils footnote" frame="void" id="id2" rules="none">
<colgroup><col class="label" /><col /></colgroup>
<tbody valign="top">
<tr><td class="label"><a class="fn-backref" href="#id2">[1]</a></td><td>C. Lu, J. Feng, S. Yan, Z. Lin. A Unified Alternating Direction Method of Multipliers by Majorization Minimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 527-541, 2018</td></tr>
</tbody>
</table>

