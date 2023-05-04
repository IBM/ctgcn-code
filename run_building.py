from ctgcn.causal_discovery import DecomposedCausalDiscovery, CausalDiscovery

DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=10, reuse_clusters=True, max_steps=61,
                                  aggregation_method='ANYW',result_path='Results/buildingheating_fixed')


DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=10, reuse_clusters=False, max_steps=61,
                                  aggregation_method='ANYW',result_path='Results/buildingheating_flex')
