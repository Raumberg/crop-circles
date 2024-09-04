import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from pandas import DataFrame
from typing import Tuple


class Network(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_first: int, hidden_dim_second: int, output_dim: int):
        """
        Model initialization with the given parameters.
        """
        super(Network, self).__init__()
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
        self.inp = nn.Linear(input_dim, hidden_dim_first)
        self.lin = nn.Linear(hidden_dim_first, hidden_dim_second) 
        self.out = nn.Linear(hidden_dim_second, output_dim)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method
        """
        x: torch.Tensor = self.relu(self.inp(x))
        x: torch.Tensor = self.relu(self.lin(x))
        x: torch.Tensor = self.drop(x)
        x: torch.Tensor = self.sigm(self.out(x))      
        return x
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, epochs: int = 100, batch_size: int = 32, learning_rate: int = 0.01):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
    
        X = torch.tensor(X_train.values, dtype=torch.float32)
        y = torch.tensor(y_train.values, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0.01)

        for epoch in range(epochs):
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

            print(f'|Epoch: {epoch+1}|, Loss --> {loss.item()}')

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X = torch.tensor(X_test, dtype=torch.float32)

        self.eval()

        with torch.no_grad():
            outputs = self(X)

        predicted_proba = torch.sigmoid(outputs)[:, 1]
        predicted = (predicted_proba > 0.5).int()

        return predicted.numpy()

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[float, float]:
        X = torch.tensor(X_test, dtype=torch.float32)
        y = torch.tensor(y_test, dtype=torch.float32)

        with torch.no_grad():
            outputs = self(X)

        predicted_proba = torch.sigmoid(outputs)[:, 1]
        predicted = (predicted_proba > 0.5).int()

        accuracy = (predicted == y).sum().item() / len(y_test)

        y_pred_proba = torch.softmax(outputs, dim=1)[:, 1].detach().numpy()
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        return accuracy, auc_roc
