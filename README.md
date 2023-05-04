# Causal Temporal Graph Convolutional Neural Networks (CTGCN)

Code provided to accompany the paper of the same name presented at FunCausal 2023. CTGCN is a tool for conducting decomposed causal inference to learn the underlying relationships in a large timeseries datasets of real-world system, which can then be used as the graph in a graph convolution neural network to improve performance on downstream forecasting task that represent the underlying causality and not just correlation.

The code was developed and tested on various datasets

For details of the code please refer to the paper https://arxiv.org/abs/2303.09634:

```
@misc{langbridge2023causal,
      title={Causal Temporal Graph Convolutional Neural Networks (CTGCN)}, 
      author={Abigail Langbridge and Fearghal O'Donncha and Amadou Ba and Fabio Lorenzi and Christopher Lohse and Joern Ploennigs},
      year={2023},
      eprint={2303.09634},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Installation Guide

The provided `requirements.txt` file details the dependencies. If you have problems with conflicting dependencies, you may need to separate the system into two virtual environments: one with the requirements up to `tigramite` for causal discovery, and the second with the remainder of the dependencies for the forecasting task.

### Usage Instructions

Example calls to CTGCN are given in the `main.py` file.

### Results

Results for each dataset presented in the paper are given in the `Results` subdirectory.
