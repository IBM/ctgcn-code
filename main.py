import numpy as np
from ctgcn.causal_discovery import DecomposedCausalDiscovery, CausalDiscovery
from ctgcn.ctgcn import CTGCN

if __name__ == '__main__':
    CTGCN('Data/Traffic_Flow/trafficflow.npy', history=12, horizon=9,
          kernel_size=3, latent_feat=[32,32],
          graph='Data/Traffic_Flow/Graphs/distance.npy', verbose=1)

    CTGCN('Data/Traffic_Flow/trafficflow.npy', history=12, horizon=9,
          kernel_size=3, latent_feat=[32,32],
          graph=None, verbose=1)