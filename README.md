# Network
Pytorch based neural network for company's tabular data binary classification of clients who are willing or not willing to pay credit.  
Not new, not hard, but working

## Try learning yourself:
```python
# Obtain your dataset and perform train/test split, having
# X_train, X_test, y_train and y_test

from net import Network

net = Network(input_dim=X_train.shape[1], hidden_dim_first=512, hidden_dim_second=256, output_dim=1)
net.train(X_train=X_train, y_train=y_train)
# Optionally you can pass batch_size, epochs and learning_rate to the train method
net.predict(X_test)
```
