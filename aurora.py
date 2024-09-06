import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch import Tensor

class AuroraSet(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Aurora(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 0.001, batch_size: int = 32):
        super(Aurora, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.data_loader = None

    def forward(self, x: Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train(self, epochs: int = 100):
        for epoch in range(epochs):
            for batch in self.data_loader:
                batch = batch.float()
                self.optimizer.zero_grad()
                outputs = self(batch)
                loss = self.criterion(outputs, batch)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def process(self, data: np.ndarray) -> Tensor:
        encoded_data = self.encoder(torch.from_numpy(data).float())
        decoded_data = self.decoder(encoded_data)
        print('Encoded Data Shape:', encoded_data.shape)
        print('Decoded Data Shape:', decoded_data.shape)
        return decoded_data

    def freight(self, dataset: pd.DataFrame) -> None:
        data = AuroraSet(dataset.values)
        self.data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
