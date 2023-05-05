from ctgcn.causal_discovery import DecomposedCausalDiscovery, CausalDiscovery

import gpytorch



with gpytorch.settings.fast_computations(True,True,True):
    with gpytorch.settings.cholesky_jitter(1):
        DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=72, decomp_clusters=10, reuse_clusters=True, max_steps=61, ci_test='GPDCTorch',
                                  aggregation_method='ANYW',result_path='Results/buildingheating_fixed_GPDCTorch')


with gpytorch.settings.fast_computations(True,True,True):
    with gpytorch.settings.cholesky_jitter(1):
        DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=10, reuse_clusters=False, max_steps=61, ci_test='GPDCTorch',
                                  aggregation_method='ANYW',result_path='Results/buildingheating_flex_GPDCTorch')
