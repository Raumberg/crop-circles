import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from tensorflow.keras.utils import Progbar

class AuroraSet(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx]

class Aurora(nn.Module):
    def __init__(self, 
                input_dim: int = 784, 
                output_dim: int = 128, 
                learning_rate: float = 0.001,
                ):
        super(Aurora, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim // 2
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data_loader = None

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def inject(self, tensor: torch.Tensor, epochs: int = 100, batch_size: int = 32):
        """
        Aurora train method, works inside Tarnn class
        """
        print(f"==================")
        print(f"[Aurora autoencoder]")
        print(f"==================")
        print(f"| Configuration |")
        print(f"| - Loss fn: {self.criterion}")
        print(f"| - Epochs: {epochs}")
        print(f"| - Batch Size: {batch_size}")
        print(f"| - Topology: ")
        print(f" ---------------- ")
        print(f"[INPUT DIM] >> {self.input_dim}")
        print(f"[OUTPUT DIM] >> {self.output_dim}")

        data_loader = DataLoader(AuroraSet(tensor), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            pbar = Progbar(target=len(data_loader), width=30)
            for i, batch in enumerate(data_loader):
                batch = batch.float()
                self.optimizer.zero_grad()
                outputs = self(batch)
                loss = self.criterion(outputs, batch)
                loss.backward()
                self.optimizer.step()
                pbar.update(i + 1, values=[("Loss", loss.item())])
            print(f'[Epoch {epoch+1}] | Loss: {loss.item():.4f}')

    def process(self, data: np.ndarray | torch.Tensor, printable: bool = False, expect: str = 'decoded') -> torch.Tensor:
        encoded_data = self.encoder(torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data)
        if expect == 'encoded':
            if printable:
                print('[Encoded Data Shape] -> ', encoded_data.shape)
            return encoded_data
        elif expect == 'decoded':
            decoded_data = self.decoder(encoded_data)
            if printable:
                print('[Decoded Data Shape] -> ', decoded_data.shape)
            return decoded_data
