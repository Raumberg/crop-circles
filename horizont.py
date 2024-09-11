import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torcheval.metrics as mx

from pandas import DataFrame, Series
import numpy as np

from tensorflow.keras.utils import Progbar

class Baum(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Baum, self).__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.inp(x))
        x = self.out(x)
        return x

class Wald(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super(Wald, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Baum(input_dim, hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(Baum(hidden_dim, hidden_dim, hidden_dim))
        self.out = Baum(hidden_dim, hidden_dim, output_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return self.sigm(x)

    def inject(self, 
            X_train: DataFrame | Series | np.ndarray, 
            y_train: DataFrame | Series | np.ndarray, 
            epochs: int = 100, 
            batch_size: int = 32, 
            learning_rate: float = 0.01,
            debug: bool = False,
            ) -> None:
        """
        Trains the neural network model on the given training data.

        Args:
            X_train (DataFrame | Series | np.ndarray): The input features of the training data.
            y_train (DataFrame | Series | np.ndarray): The target variable of the training data.
            epochs (int, optional): The number of epochs to train the model. Defaults to 100.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.01.
            debug (bool, optional): Whether to print the gradients of the model during training. Defaults to False.

        Returns:
            None, just trains the model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        X = torch.from_numpy(X_train.values if isinstance(X_train, (DataFrame | Series)) else X_train).float()
        y = torch.from_numpy(y_train.values if isinstance(y_train, (DataFrame | Series)) else y_train).float()

        dataset = torch.utils.data.TensorDataset(X, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        progbar = Progbar(target=len(data_loader), width=30)
        auc_roc = mx.BinaryAUROC()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
                batch_y = batch_y.unsqueeze(-1)
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                if debug:
                    print(f"Grads tensor: {outputs}")
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                auc_roc.update(outputs.squeeze(), batch_y.squeeze())
                progbar.update(batch_idx + 1, [("loss", loss.item())])
            avg_loss = epoch_loss / len(data_loader)
            auc_roc_value = auc_roc.compute()
            print(f"[Epoch {epoch+1}] | Loss >> {avg_loss:.4f} | AUC-ROC >> {auc_roc_value:.4f}")

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
            np.ndarray: The predicted labels or probabilities.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(X_test.values if isinstance(X_test, (DataFrame | Series)) else X_test).float()

        self.eval()

        if use_batching:
            all_outputs = []
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    outputs = self(batch_X)
                    all_outputs.append(outputs)

            all_outputs = torch.cat(all_outputs)

        else:
            with torch.no_grad():
                all_outputs = self(X)   

        if proba:
            return all_outputs.numpy()
        else:
            if len(all_outputs.shape) == 1:
                predicted = (all_outputs > 0.5).int()
            else:
                predicted = torch.argmax(all_outputs, dim=1)

            return predicted.numpy()
    
    def evaluate(self, 
        X_test: DataFrame | Series | np.ndarray, 
        y_test: DataFrame | Series | np.ndarray, 
        batch_size: int = 32) -> None:
        """
        Evaluates the model on the test data.

        Args:
            X_test (DataFrame | Series | np.ndarray): The input features of the test data.
            y_test (DataFrame | Series | np.ndarray): The target variable of the test data.
            batch_size (int, optional): The batch size for evaluation. Defaults to 32.

        Returns:
            None, just prints the evaluation metrics
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        X = torch.from_numpy(X_test.values if isinstance(X_test, (DataFrame | Series)) else X_test).float()
        y = torch.from_numpy(y_test.values if isinstance(y_test, (DataFrame | Series)) else y_test).float()

        dataset = torch.utils.data.TensorDataset(X, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        accuracy = mx.BinaryAccuracy()
        auc_roc = mx.BinaryAUROC()

        with torch.no_grad():
            self.eval()
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = self(batch_X)
                _, predicted = torch.max(outputs, 1)
                accuracy.update(predicted.squeeze(), batch_y)
                auc_roc.update(outputs.squeeze(), batch_y)

        accuracy_value = accuracy.compute()
        auc_roc_value = auc_roc.compute()
        print(f"Test Accuracy: {accuracy_value:.4f}")
        print(f"Test AUC-ROC: {auc_roc_value:.4f}")


class Horizont(nn.Module):
    def __init__(self, num_models: int, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super(Horizont, self).__init__()
        self.models = nn.ModuleList([Wald(num_layers, input_dim, hidden_dim, output_dim) for _ in range(num_models)])

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).mean(dim=0)

    def train(self, X_train, y_train, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01):
        for model in self.models:
            model.inject(X_train, y_train, epochs, batch_size, learning_rate)

    def predict(self, X_test, proba: bool = False, use_batching: bool = False, batch_size: int = 32):
        outputs = []
        for model in self.models:
            outputs.append(model.predict(X_test, proba, use_batching, batch_size))
        return np.array(outputs).mean(axis=0)

    def evaluate(self, X_test, y_test, batch_size: int = 32):
        for model in self.models:
            model.evaluate(X_test, y_test, batch_size)
