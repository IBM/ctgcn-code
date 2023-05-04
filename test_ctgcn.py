from ctgcn.causal_discovery import DecomposedCausalDiscovery, CausalDiscovery
import numpy as np

graph = DecomposedCausalDiscovery('/Users/abilangbridge/Desktop/IBM/AI4DT/CTGCN/Data/Building/buildingheating.npy',
                                  tau_max=6, decomp_period=144, decomp_clusters=3, cluster_seeding=True,
                                  aggregation_method='ANYW')

np.save(f"/Users/abilangbridge/Desktop/IBM/AI4DT/CTGCN/Graphs/Building/T6_PT144_C3_Seeding_ANYW.npy", graph)