from typing import List, Enum, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import GPDC, ParCorr, GPDCtorch

from tslearn.clustering import TimeSeriesKMeans
import warnings

import networkx as nx
import os
import json
from tqdm import tqdm

from datetime import timedelta, datetime


# arg ideas: data_path (location of data, Data/), verbose=1 (ie printing times), decomposition (time, space, both), params (tau_max/pc_alpha)
class CausalDiscovery:

    def __init__(self,
                 data_path: str,
                 tau_max: float,
                 var_names: Optional[List[str]] = None,
                 pc_alpha: Optional[float] = 0.01,
                 ci_test: Optional[Literal["GPDC", "GPDCTorch",
                                           "ParCorr"]] = 'GPDCTorch',
                 verbose: Optional[int] = 1,
                 result_path: Optional[str] = None):
        self.data_path = data_path
        self.tau_max = tau_max
        self.var_names = var_names
        self.pc_alpha = pc_alpha
        self.result_path = result_path
        if ci_test.lower() == 'gpdc':
            self.ci_test = GPDC(significance='analytic')
        if ci_test.lower() == 'gpdctorch':
            self.ci_test = GPDCtorch(significance='analytic')
        else:
            self.ci_test = ParCorr()
        self.verbose = verbose

        if self.result_path:
            os.makedirs(self.result_path, exist_ok=True)

        data = np.load(data_path)
        self.adj = self.discover(data)

    def discover(self, data):
        dataframe = pp.DataFrame(data)

        pcmci = PCMCI(dataframe=dataframe,
                      cond_ind_test=self.ci_test,
                      verbosity=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = pcmci.run_pcmciplus(tau_max=self.tau_max,
                                          pc_alpha=self.pc_alpha)

        sig_links = (results['p_matrix'] <= self.pc_alpha).astype(int)
        adj = np.sum(sig_links, axis=2)
        adj[adj > 1] = 1
        adj[adj < 1] = 0

        return adj

    def plot(self):
        if not self.var_names:
            self.var_names = [x for x in range(len(self.adj))]

        mapping = {i: str(x) for i, x in enumerate(self.var_names)}
        FCG = nx.from_numpy_matrix(
            np.ones((len(self.var_names), len(self.var_names))))
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

        plt.figure(figsize=(15, 10))
        nx.draw(G, node_size=10, edge_color="tab:blue", pos=pos)


class DecomposedCausalDiscovery(CausalDiscovery):

    def __init__(self,
                 data_path: str,
                 tau_max: float,
                 var_names: Optional[List[str]] = None,
                 pc_alpha: Optional[float] = 0.01,
                 ci_test: Optional[Literal["GPDC", "GPDCTorch",
                                           "ParCorr"]] = 'GPDCTorch',
                 verbose: Optional[int] = 1,
                 result_path: Optional[str] = None,
                 decomp_period: Optional[int] = None,
                 decomp_clusters: Optional[bool] = None,
                 reuse_clusters: Optional[bool] = False,
                 cluster_centers: Optional[List[List[float]]] = None,
                 aggregation_method: Optional[Literal['MTW', 'MTU', 'ANYU',
                                                      'SUM']] = 'MTW',
                 max_steps: Optional[int] = None):
        self.data_path = data_path
        self.tau_max = tau_max
        self.var_names = var_names
        self.pc_alpha = pc_alpha
        self.max_steps = max_steps
        self.result_path = result_path
        if ci_test.lower() == 'gpdc':
            self.ci_test = GPDC(significance='analytic')
        else:
            self.ci_test = ParCorr()
        self.verbose = verbose

        if self.result_path:
            os.makedirs(self.result_path, exist_ok=True)

        self.decomp_period = decomp_period
        self.decomp_clusters = decomp_clusters
        self.reuse_clusters = reuse_clusters
        self.cluster_centers = cluster_centers

        self.aggregation_method = aggregation_method
        self.starttime = datetime.now()

        data = np.load(data_path)
        self.adj = self.decompose(data)
        self.save()

    def decompose(self, data):
        assert (
            self.decomp_period or self.decomp_clusters
        ), "DecomposedCausalDiscovery must conduct at least one decomposition.\
                                                              For un-decomposed causal discovery, use the CausalDiscovery class."

        self.adj_list = []
        self.stats = {
            "var_names": self.var_names,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "starttime": self.starttime.isoformat(),
            "decomp_period": self.decomp_period,
            "decomp_clusters": self.decomp_clusters,
            "reuse_clusters": self.reuse_clusters,
            "max_steps": self.max_steps,
            "aggregation_method": self.aggregation_method,
            "data_path": self.data_path,
            "result_path": self.result_path,
            "steps": []
        }

        if self.decomp_period:
            assert self.decomp_period < len(
                data
            ), "The decomposition period must be smaller than the length of the time axis."
            periods = [
                i * self.decomp_period
                for i in range(len(data) // self.decomp_period + 1)
            ]
        else:
            periods = [0, len(data)]

        if self.decomp_clusters:
            assert self.decomp_clusters < len(
                data[0]
            ), "The number of clusters must be smaller than the number of variables."

        # Warm up clustering
        if self.decomp_clusters and self.reuse_clusters and self.cluster_centers is None:
            print(f'Warming up clusters')
            for i, (start, end) in enumerate(tqdm(zip(periods, periods[1:]))):
                if i == 0:
                    km = TimeSeriesKMeans(n_clusters=self.decomp_clusters,
                                          metric="dtw",
                                          n_jobs=-1)
                else:
                    km = TimeSeriesKMeans(n_clusters=self.decomp_clusters,
                                          metric="dtw",
                                          init=km.cluster_centers_,
                                          n_jobs=-1)
                km.fit_predict(data[start:end, :].transpose())

                if self.max_steps and i >= self.max_steps:
                    if self.verbose > 0:
                        print(f'Max steps reached. Stopping causal discovery.')
                    break

        for i, (start, end) in enumerate(tqdm(zip(periods, periods[1:]))):
            steptime = datetime.now()
            if not self.decomp_clusters:
                print(f'Temporal split {i} of {len(periods)-1}.')
                adj = self.discover(data[start:end, :])
                self.adj_list.append(adj)
            else:
                adj = np.zeros((len(data[0]), len(data[0])))
                if self.reuse_clusters and self.cluster_centers is not None:
                    km = TimeSeriesKMeans(n_clusters=self.decomp_clusters,
                                          metric="dtw",
                                          init=self.cluster_centers,
                                          n_jobs=-1)
                elif self.reuse_clusters:
                    km = TimeSeriesKMeans(n_clusters=self.decomp_clusters,
                                          metric="dtw",
                                          init=km.cluster_centers_,
                                          n_jobs=-1)
                else:
                    km = TimeSeriesKMeans(n_clusters=self.decomp_clusters,
                                          metric="dtw",
                                          n_jobs=-1)
                y_pred = km.fit_predict(data[start:end, :].transpose())

                if self.verbose > 0:
                    print(
                        f'\nTemporal split {i} of {len(periods)-1} has clustering inertia: {km.inertia_}'
                    )

                cluster_dict = {
                    x: y
                    for x, y in zip([x for x in range(len(data[0]))], y_pred)
                }

                cluster_map = {}
                for k, v in cluster_dict.items():
                    cluster_map[v] = cluster_map.get(v, []) + [k]

                for cluster, idxs in cluster_map.items():
                    if len(idxs) == 1:
                        continue

                    if self.verbose > 0:
                        print(
                            f'Running causal discovery on cluster {cluster} with {len(idxs)} nodes.'
                        )

                    sub_data = data[start:end, idxs]
                    sub_adj = self.discover(sub_data)

                    for i, r1 in enumerate(idxs):
                        for j, r2 in enumerate(idxs):
                            adj[r1, r2] = sub_adj[i, j]

            dtnow = datetime.now()
            self.stats["steps"].append({
                "step":
                i,
                "window_start":
                start,
                "window_end":
                end,
                "adj":
                adj.tolist(),
                "cluster_centers":
                km.cluster_centers_.tolist(),
                "clusters":
                list(cluster_map.values()),
                "runtime": (dtnow - steptime).total_seconds(),
                "starttime":
                steptime.isoformat(),
                "endtime":
                dtnow.isoformat(),
            })
            self.stats["runtime_total"] = (dtnow -
                                           self.starttime).total_seconds()
            self.stats["endtime"] = dtnow.isoformat()

            if self.result_path:
                with open(os.path.join(self.result_path, f"stats.json"),
                          "wt") as fo:
                    json.dump(self.stats, fo)

            self.adj_list.append(adj)

            if self.max_steps and i >= self.max_steps:
                if self.verbose > 0:
                    print(f'Max steps reached. Stopping causal discovery.')
                break

        return self.aggregate()

    def aggregate(self, aggregation_method=None):
        if len(self.adj_list) == 1:
            return self.adj_list[0]
        if aggregation_method is None:
            aggregation_method = self.aggregation_method

        adj = np.sum(self.adj_list, axis=0)

        if aggregation_method == 'MTW':
            return np.where(adj >= np.max(adj) * 0.5, adj, 0)
        elif aggregation_method == 'MTU':
            return np.where(adj >= np.max(adj) * 0.5, 1, 0)
        elif aggregation_method == 'ANYU':
            return np.where(adj > 0, 1, 0)
        else:
            return adj

    def save(self, result_path: Optional[str] = None):
        self.stats["adj_MTW"] = self.aggregate('MTW').tolist()
        self.stats["adj_MTU"] = self.aggregate('MTU').tolist()
        self.stats["adj_ANYU"] = self.aggregate('ANYU').tolist()
        self.stats["adj_SUM"] = self.aggregate('SUM').tolist()
        if result_path is None:
            result_path = self.result_path

        if result_path:
            np.save(os.path.join(result_path, f"results.np"), self)
            with open(os.path.join(result_path, f"stats.json"), "wt") as fo:
                json.dump(self.stats, fo)
