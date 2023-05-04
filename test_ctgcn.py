from ctgcn.causal_discovery import DecomposedCausalDiscovery, CausalDiscovery

graph = DecomposedCausalDiscovery('Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=3, reuse_clusters=True,
                                  aggregation_method='ANYW')