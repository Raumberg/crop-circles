import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pandas import DataFrame, Series
from typing import Tuple

try:
    import IPython
    if IPython.get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except (ImportError, NameError):
    from tqdm import tqdm

from rich import print
from rich.console import Console
from rich.table import Table

console = Console()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
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
        x: torch.Tensor = self.drop(x)
        x: torch.Tensor = self.relu(self.lin(x))
        x: torch.Tensor = self.sigm(self.out(x))
        return x
    
    def train(self, 
            X_train: DataFrame | Series | np.ndarray, 
            y_train: DataFrame | Series | np.ndarray, 
            epochs: int = 100, 
            batch_size: int = 32, 
            learning_rate: int = 0.01,
            debug: bool = False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X = torch.from_numpy(X_train.values if isinstance(X_train, (DataFrame | Series)) else X_train).float()
        y = torch.from_numpy(y_train.values if isinstance(y_train, (DataFrame | Series)) else y_train).float()

        dataset = torch.utils.data.TensorDataset(X, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0.01)

        console.print(f"[bold]Lake model[/bold]")
        console.print(f"[bold]-------------------------[/bold]")
        console.print(f"[bold green]Device:[/bold green] {device}")
        console.print(f"[bold green]Epochs:[/bold green] {epochs}")
        console.print(f"[bold green]Batch Size:[/bold green] {batch_size}")
        console.print(f"[bold green]Learning Rate:[/bold green] {learning_rate}")
        console.print(f"[bold]-------------------------[/bold]")

        for epoch in range(epochs):
            epoch_loss = 0
            table = Table(title=f"Epoch {epoch+1}")
            table.add_column("Batch", justify="right", style="cyan")
            table.add_column("Loss", justify="right", style="magenta")
            with tqdm(data_loader, 
            desc=f'|Epoch {epoch+1}|', 
            unit='batches', 
            ) as pbar:
                for batch_X, batch_y in pbar:
                    batch_y = batch_y.unsqueeze(-1)
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    if debug:
                        console.print(f"[bold]Grads tensor:[/bold] {outputs}")
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            avg_loss = epoch_loss / len(data_loader)
            console.print(table)
            console.print(f"[bold]Epoch {epoch+1}, Average Loss:[/bold] {avg_loss:.4f}")
            console.print("-------------------------")

    def predict(self, X_test: DataFrame) -> np.ndarray:
        X = torch.tensor(X_test, dtype=torch.float32)

        self.eval()

        with torch.no_grad():
            outputs = self(X)

        predicted_proba = torch.sigmoid(outputs)[:, 1]
        predicted = (predicted_proba > 0.5).int()

        return predicted.numpy()

    def evaluate(self, X_test: DataFrame, y_test: DataFrame) -> Tuple[float, float]:
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
