import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import warnings

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric_temporal.signal.static_graph_temporal_signal as SGTS
from torch_geometric_temporal.signal import temporal_signal_split

from tslearn.preprocessing import TimeSeriesScalerMinMax
from scipy import sparse

from ctgcn.causal_discovery import CausalDiscovery, DecomposedCausalDiscovery

class CTGCN:
    def __init__(self, data_path, history, horizon, rescale=False, train_ratio=0.66, val_ratio=0.33, epochs=50, batch_size=32, kernel_size=2, latent_feat=[64,32], patience=5, min_delta=0.1, lr=0.005, graph=None, discovery_params=None, verbose=1):
        self.history = history
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.latent_feat = latent_feat
        self.patience = patience
        self.min_delta = min_delta
        self.lr = lr

        if not graph:
            assert discovery_params, "If a graph is not supplied, parameters for causal discovery must be specified."
            self.graph = DecomposedCausalDiscovery(data_path, **discovery_params)
        else:
            self.graph = np.load(graph)

        self.verbose = verbose

        data = np.load(data_path)
        self.train(data)

    def create_dataset(self, data):
        if self.rescale:
            data = TimeSeriesScalerMinMax(value_range=(0,1)).fit_transform(data)
        X = np.zeros((data.shape[0]-self.history, data.shape[1], self.history))
        Y = np.zeros((data.shape[0]-self.history, data.shape[1], self.horizon))

        for i in range(data.shape[0]-self.history):
            X[i,:,:] = data[i:i+self.history, :].T

        for i in range(X.shape[0]-self.history):
            Y[i,:] = X[i+self.history,:,:self.horizon]
        X = X[:-self.history,:]
        Y = Y[:-self.history,:]

        A = self.graph + np.identity(self.graph.shape[0])
        degrees = np.power(np.array(A.sum(1)), -0.5).ravel()
        degrees[np.isinf(degrees)] = 0.0
        D = np.diag(degrees)
        A = A.dot(D.T).dot(D)

        adj_coo = sparse.coo_matrix(A)
        row = torch.tensor(adj_coo.row)
        col = torch.tensor(adj_coo.col) 
        edge_attr = torch.tensor(adj_coo.data).type(torch.float)
        edge_index = torch.stack([row, col], dim=0).type(torch.long)

        return SGTS.StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_attr, features=X, targets=Y)

    def train(self, data):
        dataset = self.create_dataset(data)

        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=self.train_ratio)
        val_dataset, test_dataset = temporal_signal_split(test_dataset, train_ratio=self.val_ratio)

        if self.verbose > 0:
            print(f"Train:Val:Test = {len(list(train_dataset))}:{len(list(val_dataset))}:{len(list(test_dataset))}")

        model = TGCN(self.history, self.horizon, self.kernel_size, self.latent_feat)
        stopper = EarlyStopper(self.patience, self.min_delta)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs): 
            loss = 0
            for time, snapshot in enumerate(train_dataset):
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                loss += torch.mean((y_hat-snapshot.y)**2)

                if time % self.batch_size == self.batch_size - 1:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0

            with torch.no_grad():
                val_loss = 0
                for time, val_snapshot in enumerate(val_dataset):
                    val_hat = model(val_snapshot.x, val_snapshot.edge_index, val_snapshot.edge_attr)
                    val_loss += torch.mean((val_hat-val_snapshot.y)**2)

                if stopper.early_stop(val_loss/(time+1)):
                    if self.verbose > 0:
                        print("Early stopping!")             
                    break

                if self.verbose > 0:
                    print(f"Epoch {epoch} RMSE {np.sqrt(val_loss / (time+1)):.5f}")

        if self.verbose > 0:
            print(f"Training complete.\n")
        
        self.forecast(model, data, test_dataset)

    def forecast(self, model, data, test_dataset):
        model.eval()
        rmse = 0

        predictions = np.zeros((len(list(test_dataset)), len(data[0]), self.horizon))
        labels = np.zeros((len(list(test_dataset)), len(data[0]), self.horizon))
        inputs = np.zeros((len(list(test_dataset)), len(data[0]), self.history))

        with torch.no_grad():
            for time, snapshot in enumerate(test_dataset):
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

                rmse = rmse + abs(torch.mean((y_hat-snapshot.y)**2))
                
                predictions[time, :, :] = y_hat.detach().numpy()
                labels[time, :, :] = snapshot.y
                inputs[time, :, :] = snapshot.x

        rmse = rmse / (time+1)
        rmse = np.sqrt(rmse.item())

        print(f"Forecast RMSE:\t{rmse:.6f}")

class TGCN(torch.nn.Module):
    def __init__(self, node_features, periods, kernel_size, latent_features):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(1, kernel_size))
        self.gcn1 = torch_geometric.nn.GCNConv(node_features-kernel_size+1, latent_features[0])
        self.gcn2 = torch_geometric.nn.GCNConv(latent_features[0], latent_features[1])
        self.linear = torch.nn.Linear(latent_features[1], periods)

    def forward(self, x, edge_index, edge_weights):
        x = torch.unsqueeze(x, dim=0)
        x = self.conv1(x)
        x = F.relu(x)
        x = x[0,:,:]

        x = self.gcn1(x, edge_index, edge_weights)
        x = F.relu(x)

        x = self.gcn2(x, edge_index, edge_weights)
        x = F.relu(x)

        x = self.linear(x)
        return x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False