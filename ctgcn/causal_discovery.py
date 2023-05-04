import numpy as np
import matplotlib.pyplot as plt

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import GPDC, GPDCtorch, ParCorr

from tslearn.clustering import TimeSeriesKMeans
import warnings

import networkx as nx

# arg ideas: data_path (location of data, Data/), verbose=1 (ie printing times), decomposition (time, space, both), params (tau_max/pc_alpha)
class CausalDiscovery:
    def __init__(self, data_path, tau_max, var_names=None, pc_alpha=0.01, ci_test='gpdc', verbose=1):
        self.data_path = data_path
        self.tau_max = tau_max
        self.var_names = var_names
        self.pc_alpha = pc_alpha
        if ci_test.lower() == 'gpdc':
            self.ci_test = GPDC(significance='analytic')
        elif ci_test.lower() == 'gpdctorch':
            self.ci_test = GPDCtorch(significance='analytic')
        else:
            self.ci_test = ParCorr()
        self.verbose = verbose

        data = np.load(data_path)
        self.adj = self.discover(data)

    def discover(self, data):
        dataframe = pp.DataFrame(data)

        pcmci = PCMCI(dataframe = dataframe, 
                      cond_ind_test = self.ci_test,
                      verbosity = 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = pcmci.run_pcmciplus(tau_max=self.tau_max, pc_alpha=self.pc_alpha)

        sig_links = (results['p_matrix'] <= self.pc_alpha).astype(int)
        adj = np.sum(sig_links, axis=2)
        adj[adj > 1] = 1
        adj[adj < 1] = 0

        return adj

    def plot(self):
        if not self.var_names:
            self.var_names = [x for x in range(len(self.adj))]

        mapping =  {i: str(x) for i, x in enumerate(self.var_names)}
        FCG = nx.from_numpy_matrix(np.ones((len(self.var_names), len(self.var_names))))
        FCG = nx.relabel_nodes(FCG, mapping)
        pos = nx.spring_layout(FCG, iterations=10, seed=100)

        adj = (adj + adj.T >= 1)
        adj = adj - np.eye(adj.shape[0])
        adj[adj < 1] = 0

        G = nx.from_numpy_matrix(adj)
        G = nx.relabel_nodes(G, mapping)

        if self.verbose > 0:
            print(f"Graph density: {nx.density(G)}")
            print(f"Number of edges: {len(G.edges)}")

        plt.figure(figsize=(15,10))
        nx.draw(G, node_size=10, edge_color="tab:blue", pos=pos)


class DecomposedCausalDiscovery(CausalDiscovery):
    def __init__(self, data_path, tau_max, var_names=None, pc_alpha=0.01, ci_test='gpdc', decomp_period=None, decomp_clusters=None, cluster_seeding=False, aggregation_method='MTW', verbose=1):
        self.data_path = data_path
        self.tau_max = tau_max
        self.var_names = var_names
        self.pc_alpha = pc_alpha
        if ci_test.lower() == 'gpdc':
            self.ci_test = GPDC(significance='analytic')
        elif ci_test.lower() == 'gpdctorch':
            self.ci_test = GPDCtorch(significance='analytic')
        else:
            self.ci_test = ParCorr()
        self.verbose = verbose

        self.decomp_period = decomp_period
        self.decomp_clusters = decomp_clusters
        self.cluster_seeding = cluster_seeding

        self.aggregation_method = aggregation_method

        data = np.load(data_path)
        self.adj = self.decompose(data)

    def decompose(self, data):
        assert (self.decomp_period or self.decomp_clusters), "DecomposedCausalDiscovery must conduct at least one decomposition.\
                                                              For un-decomposed causal discovery, use the CausalDiscovery class."
        
        adj_list = []

        if self.decomp_period:
            assert self.decomp_period < len(data), "The decomposition period must be smaller than the length of the time axis."
            periods = [i*self.decomp_period for i in range(len(data)//self.decomp_period + 1)]
        else:
            periods = [0, len(data)]

        for i, (start, end) in enumerate(zip(periods, periods[1:])):
            if not self.decomp_clusters:
                print(f'Temporal split {i} of {len(periods)-1}.')
                adj = self.discover(data[start:end, :])
                adj_list.append(adj)

            else:
                assert self.decomp_clusters < len(data[0]), "The number of clusters must be smaller than the number of variables."

                adj = np.zeros((len(data[0]), len(data[0])))
                if self.cluster_seeding and i > 0:
                    km = TimeSeriesKMeans(n_clusters=self.decomp_clusters, metric="dtw", init=km.cluster_centers_, n_jobs=-1)
                else:
                    km = TimeSeriesKMeans(n_clusters=self.decomp_clusters, metric="dtw", n_jobs=-1)
                y_pred = km.fit_predict(data[start:end, :].transpose())

                if self.verbose > 0:
                    print(f'Temporal split {i} of {len(periods)-1} has clustering inertia: {km.inertia_}')

                cluster_dict = {x: y for x, y in zip([x for x in range(len(data[0]))], y_pred)}
                cluster_map = {}
                for k, v in cluster_dict.items():
                    cluster_map[v] = cluster_map.get(v, []) + [k]

                for cluster, idxs in cluster_map.items():
                    if len(idxs) == 1:
                        continue

                    if self.verbose > 0:
                        print(f'Running causal discovery on cluster {cluster} with {len(idxs)} nodes.')
                    
                    sub_data = data[start:end, idxs]
                    sub_adj = self.discover(sub_data)

                    for i, r1 in enumerate(idxs):
                        for j, r2 in enumerate(idxs):
                            adj[r1, r2] = sub_adj[i, j]
                adj_list.append(adj)
        return self.aggregate(adj_list)

    def aggregate(self, adj_list):
        if len(adj_list) == 1:
            return adj_list[0]

        adj = np.sum(adj_list, axis=0)

        if self.aggregation_method == 'MTW':
            return np.where(adj >= np.max(adj)*0.5, adj, 0)
        elif self.aggregation_method == 'MTU':
            return np.where(adj >= np.max(adj)*0.5, 1, 0)
        elif self.aggregation_method == 'ANYU':
            return np.where(adj > 0, 1, 0)
        else:
            return adj
        

