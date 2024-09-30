# Network
Pytorch based neural network for company's tabular data binary classification of clients who are willing or not willing to pay credit.  
Not new, not hard, but working
### Contents:
* *tandem*       -> main neural network class
* *riegel*   ->  embedding model for tabular categorical variables (with additional attention mechanism)
* *aurora*   -> autoencoder model for tabular categorical variables, reducing dimentionality
* *superforest* -> deep neural decision forest interpretation (simplified)
* *data* -> dataminer helper class for manipulating dataframes

## Try learning yourself:
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
