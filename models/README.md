## 💻 GUIDELINES
### If you are trying to run models yourself:
**Note:**  
This guideline only utilizes the main model TANDEM, omitting aurora and riegel usage
```python
# Obtain your dataset and perform train/test split, having
# X_train, X_test, y_train and y_test
# You can use DataMiner from data.py:
from data import DataMiner

df = pd.read_csv('..')
miner = DataMiner(dataframe=df)
X_train, X_test, y_train, y_test = miner.split(test_size=0.2, random_state=42, use_val=False)  
# You can use <True> boolean flag for <use_val> to create a validation set.
# In that case, you can also provide <val_size>

from net import TANDEM

net = TANDEM(
  input_dim=X_train.shape[1],
  hidden_dim=512,
  output_dim=1,
  method='regression',
  dropout=False,
  loss='mse',
  aurora=False
)
net.inject(
  X_train,
  y_train,
  batch_size=256,
  epochs=50,
  use_tqdm=False,
  learning_rate=0.01,
  clip_grads=False,
  debug=False
)
preds = net.predict(X_test)
net.save(path='../data/')
net.inspect_model()
net.params
```
### An incomplete (for 2024.10.07) guide for Riegel Ring usage:
```python
import torch 

# Obtain data and cast to tensors:
X_train_tensor = torch.tensor(X_train.values)
y_train_tensor = torch.tensor(y_train.values)
X_test_tensor = torch.tensor(X_test.values)
y_test_tensor = torch.tensor(y_test.values)

# function to see what parameters need weight decay:
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

# setting up dataloaders, optimizer with parameters and loss function (in this case, binary xentropy)
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True
)
parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
params = [
        {'params': parameters_with_wd},
        {'params': parameters_without_wd, 'weight_decay': 0.0},
]
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=params, lr=0.001)

# defining the model:
import riegel as rr

model = rr.RiegelRing(
    d_numerical=50,
    categories=None,
    token_bias=True,
    n_layers=3,
    d_token=256,
    n_heads=32,
    attention_dropout=0.3,
    ffn_dropout=0.,
    residual_dropout=0.0,
    prenormalization=False,
    d_out=1,
    kv_compression=None,
    kv_compression_sharing=None,
)

# train the model manually:
for epoch in range(1, 50):
    model.train()
    for iteration, batch in enumerate(train_loader):
        inputs, targets = batch
        targets = targets.float().squeeze()
        optimizer.zero_grad()
        loss = loss_fn(model(batch), targets)
        loss.backward()
        optimizer.step()

# ---- Alternatively ----
# You can use in-box methods (which may be in developing state) of Riegel Ring class with cute Tensorflow's progress bar:
model = rr.RiegelRing(**parameters).setup()
model.inject(
  loader=train_loader
  epochs=50
)
```
