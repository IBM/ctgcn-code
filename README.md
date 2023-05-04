# Causal Temporal Graph Convolutional Neural Networks (CTGCN)

Code provided to accompany the paper of the same name submitted to ICML 2023. CTGCN is a tool for conducting decomposed causal inference to learn the underlying relationships in a real-world system, which can then be used as the graph in a graph convolution layer to improve performance on some downstream forecasting task.

### Installation Guide

The provided `requirements.txt` file details the dependencies. If you have problems with conflicting dependencies, you may need to separate the system into two virtual environments: one with the requirements up to `tigramite` for causal discovery, and the second with the remainder of the dependencies for the forecasting task.

### Usage Instructions

Example calls to CTGCN are given in the `main.py` file.

### Results

Results for each dataset presented in the paper are given in the `Results` subdirectory.
