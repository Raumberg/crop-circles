# Inspired and based on arxiv article "Well-tuned Simple Nets Excel on Tabular Datasets"
# by Arlind Kadra, Marius Lindauer, Frank Hutter and Josif Grabocka
# Link: https://arxiv.org/pdf/2106.11189

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as nn_init  
import math
from torchinfo import summary

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

from pandas import DataFrame, Series
from typing import Tuple

from lakes.aurora import Aurora

from tensorflow.keras.utils import Progbar

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"[Module imported. Init device: {DEVICE}]")


class TANDEM(nn.Module):
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                output_dim: int, 
                method: str = 'classification', 
                dropout: float = 0.0, 
                loss: str = 'bce',
                aurora: bool = False,
                shape: int = None,
                weight_noise_std: float = 0.0,
                gradient_noise_std: float = 0.0
                ):
        """
        Temporal Abberated Neural Differentiable Embedding Mechanism [TANDEM]
        Model initialization with the given parameters.
        """
        super(TANDEM, self).__init__()
        assert input_dim is not None, 'Please, provide the initial shape of X_train using X_train.shape[1]'
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, hidden_dim // 2) 
        self.out = nn.Linear(hidden_dim // 2, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.method = method
        self.use_dropout = dropout
        self.loss_fn = self.set_loss(loss)

        if aurora:
            self.aurora = Aurora(input_dim=shape, learning_rate=0.01, output_dim=10)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._attenuated_kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero

        self.weight_noise_std = weight_noise_std  # Initialize weight noise std
        self.gradient_noise_std = gradient_noise_std  # Initialize gradient noise std        

    def _attenuated_kaiming_uniform_(self, tensor, a=math.sqrt(5), scale=1., mode='fan_in', nonlinearity='relu'):
        """
        Initializes a tensor using an attenuated Kaiming uniform distribution.
        """
        fan = nn_init._calculate_correct_fan(tensor, mode)
        gain = nn_init.calculate_gain(nonlinearity, a)
        std = gain * scale / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def inject_weight_noise(self):
        """
        Injection of weight noise to weight params according to std provided
        """
        for param in self.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * self.weight_noise_std
                param.data += noise

    def inject_gradient_noise(self):
        """
        Injection of grad noise to grad params according to std provided
        """
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                noise = torch.randn_like(param.grad) * self.gradient_noise_std
                param.grad += noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method
        """
        if hasattr(self, 'aurora'):
            x_encoded = self.aurora.process(x)
            x = x_encoded
        x = self.relu(self.bn1(self.inp(x)))
        x = self.drop(x)
        x = self.relu(self.bn2(self.lin(x)))
        x = self.out(x)
        return x
    
    def set_loss(self, loss):
        match loss:
            case 'bce': return nn.BCEWithLogitsLoss()
            case 'mse': return nn.MSELoss()
            case 'l1': return nn.L1Loss()
            case 'sl1': return nn.SmoothL1Loss()
            case 'poisson': return nn.PoissonNLLLoss()
            case 'gaussian': return nn.GaussianNLLLoss()
            case 'kldiv': return nn.KLDivLoss()
            case _: raise ValueError("Invalid loss. Available losses - 'bce', 'mse', 'l1', 'sl1', 'poisson', 'gaussian', 'kldiv'.")
        
    def inject(self, 
            X_train: DataFrame | Series | np.ndarray, 
            y_train: DataFrame | Series | np.ndarray, 
            epochs: int = 100, 
            batch_size: int = 32, 
            learning_rate: float = 0.01,
            debug: bool = False,
            clip_grads: bool = False,
            ) -> None:
        """
        Trains the neural network model on the given training data.

        Args:
            X_train (DataFrame | Series | np.ndarray): The input features of the training data.
            y_train (DataFrame | Series | np.ndarray): The target variable of the training data.
            epochs (int, optional): The number of epochs to train the model. Defaults to 100.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.01.
            debug (bool, optional): Whether to print the gradients of the model during training. Defaults to False
            use_tqdm (bool, optional): Whether to use the tqdm progress bar during training. Defaults to False, using tensorflow's progress bar. 
            clip_grads (bool, optional): Whether to clip gradient norms during the training phase. Defaults to False.

        Returns:
            None, just trains the model
        """
        self.to(DEVICE)

        X = torch.from_numpy(X_train.values if isinstance(X_train, (DataFrame | Series)) else X_train).float()
        y = torch.from_numpy(y_train.values if isinstance(y_train, (DataFrame | Series)) else y_train).float()

        if hasattr(self, 'aurora'):
            self.fit_aurora(tensor=X)
            X_encoded = self.aurora.process(X, expect='encoded')
            dataset = torch.utils.data.TensorDataset(X_encoded, y)
        else:
            print("Aurora is not enabled, skipping fitting..\n")
            dataset = torch.utils.data.TensorDataset(X, y)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = self.loss_fn
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0.01)

        print(f"==================")
        print(f"     [TANDEM]")
        print(f"==================")
        print(f"| Configuration |")
        print(f"| - Method: {self.method}")
        print(f"| - Loss fn: {criterion}")
        print(f"| - Device: {device}")
        print(f"| - Epochs: {epochs}")
        print(f"| - Batch Size: {batch_size}")
        print(f"| - Learning Rate: {learning_rate}")
        print(f" ---------------- ")

        self.train()

        for epoch in range(epochs):
            epoch_loss = 0
            pbar = Progbar(target=len(data_loader), width=30)
            for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
                batch_y = batch_y.unsqueeze(-1)
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                if debug:
                    print(f"Grads: {outputs}")
                loss = criterion(outputs, batch_y)
                loss.backward()
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                self.inject_gradient_noise()
                optimizer.step()
                self.inject_weight_noise()

                epoch_loss += loss.item()
                pbar.update(batch_idx + 1, [("loss", loss.item())])

            avg_loss = epoch_loss / len(data_loader)
            print(f"[Epoch {epoch+1}] | Loss >> {avg_loss:.4f}")

    def predict(self, 
                X_test: DataFrame | Series | np.ndarray, 
                proba: bool = False, 
                use_batching: bool = False, 
                batch_size: int = 32) -> np.ndarray:
        """
        Makes predictions on the input data.

        Args:
            X_test (DataFrame | Series | np.ndarray): The input features.
            batch_size (int, optional): The batch size for prediction. Defaults to 32.
            use_batching (bool, optional): Whether to use batch prediction.
            proba (bool, optional): Whether to return probabilities instead of class labels. Defaults to False.

        Returns:
            np.ndarray: The predicted labels or probabilities if method == 'classification', prediction values if method == 'regression'.
        """
        X = torch.from_numpy(X_test.values if isinstance(X_test, (DataFrame | Series)) else X_test).float()

        self.eval()

        if use_batching:
            all_outputs = []
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    outputs = self(batch_X)
                    if self.method == 'classification':
                        all_outputs.append(torch.sigmoid(outputs))
                    else:
                        all_outputs.append(outputs)

            all_outputs = torch.cat(all_outputs)
        else:
            with torch.no_grad():
                all_outputs = self(X)   
                if self.method == 'classification':
                    all_outputs = torch.sigmoid(all_outputs)
                else:
                    all_outputs = self(X)

        return all_outputs.numpy()

    def evaluate(self, 
                X_test: DataFrame | Series | np.ndarray, 
                y_test: DataFrame | Series | np.ndarray,
                use_batching: bool = False,
                batch_size: int = 32) -> Tuple[float, float]:
        """
        Evaluates the model on the test data.

        Args:
            X_test (DataFrame | Series | np.ndarray): The input features of the test data.
            y_test (DataFrame | Series | np.ndarray): The target variable of the test data.
            batch_size (int, optional): The batch size for evaluation. Defaults to 32.
            use_batching (bool, optional): Whether to use batching during evaluation. Defaults to True.

        Returns:
            float, float: The Accuracy and AUC/ROC score of the model on the test data if method == 'classification', MSE and MAE if method == 'regression'.
        """
        X = torch.from_numpy(X_test.values if isinstance(X_test, (DataFrame | Series)) else X_test).float()
        y = torch.from_numpy(y_test.values if isinstance(y_test, (DataFrame | Series)) else y_test).float()

        if use_batching:
            total_outputs = []
            total_labels = []
            with torch.no_grad():
                self.eval()
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_y = y[i:i+batch_size]

                    outputs = self(batch_X)

                    total_outputs.extend(outputs.detach().numpy())
                    total_labels.extend(batch_y.numpy())

            if self.method == 'classification':
                predicted = (torch.tensor(total_outputs) > 0.5).int()
                accuracy = (predicted == torch.tensor(total_labels).unsqueeze(-1)).sum().item() / len(y_test)
                auc_roc = roc_auc_score(total_labels, total_outputs)
                return accuracy, auc_roc
            else:
                mse = mean_squared_error(total_labels, total_outputs)
                mae = mean_absolute_error(total_labels, total_outputs)
                return mse, mae
        else:
            with torch.no_grad():
                self.eval()
                outputs = self(X)

            if self.method == 'classification':
                predicted = (outputs > 0.5).int()
                accuracy = (predicted == y.unsqueeze(-1)).sum().item() / len(y_test)
                auc_roc = roc_auc_score(y_test, outputs.detach().numpy())
                return accuracy, auc_roc
            else:
                mse = mean_squared_error(y_test, outputs.detach().numpy())
                mae = mean_absolute_error(y_test, outputs.detach().numpy())
                return mse, mae

    def save(self, path: str) -> None:
        """
        Saves the model's state to a file.

        Args:
            path (str): The path to the file containing the model's state.

        Returns:
            None, just saves the model
        """
        torch.save(self.state_dict(), f"{path}_nl.pth")

    def load(self, path: str) -> None:
        """
        Loads the model's state from a file.

        Args:
            path (str): The path to the file containing the model's state.

        Returns:
            None, just loads the model
        """
        self.load_state_dict(torch.load(path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
        self.eval()

    def fit_aurora(self, tensor: torch.Tensor, epochs: int = 5, batch_size: int = 32):
        self.aurora.inject(tensor=tensor, epochs=epochs, batch_size=batch_size)

    def inspect_model(self):
        print(self)

    def inspect_params(self):
        for name, param in self.named_parameters():
            print(f"Parameter {name}, Shape: {param.shape}")

    def inspect_weights(self):
        for name, param in self.named_parameters():
            print(f"Weight {name}, Shape: {param.shape}, Values: {param.data}")

    def inspect_grads(self):
        for name, param in self.named_parameters():
            print(f"Gradient {name}, Shape: {param.shape}, Values: {param.grad.data}")

    @property
    def params(self):
        return sum(parameter.numel() for parameter in self.parameters if parameter.requires_grad)
    
    @property
    def summary(self):
        return summary(self)
