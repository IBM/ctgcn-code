from ctgcn.causal_discovery import DecomposedCausalDiscovery, CausalDiscovery

import gpytorch

with gpytorch.settings.fast_computations(True,True,True):
    with gpytorch.settings.cholesky_jitter(1):
        DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=10, one_cluster=True, ci_test='ParCorr',
                                  aggregation_method='ANYW',result_path='Results/buildingheating_single10')


with gpytorch.settings.fast_computations(True,True,True):
    with gpytorch.settings.cholesky_jitter(1):
        DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=5, one_cluster=True, ci_test='ParCorr',
                                  aggregation_method='ANYW',result_path='Results/buildingheating_single5')

with gpytorch.settings.fast_computations(True,True,True):
    with gpytorch.settings.cholesky_jitter(1):
        DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=5, reuse_clusters=True, ci_test='ParCorr',
                                  aggregation_method='ANYW',result_path='Results/buildingheating_fixed5')


with gpytorch.settings.fast_computations(True,True,True):
    with gpytorch.settings.cholesky_jitter(1):
        DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=5, reuse_clusters=False, ci_test='ParCorr',
                                  aggregation_method='ANYW',result_path='Results/buildingheating_flex5')
