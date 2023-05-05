import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

class NNPolicy:

    def __init__(self, net_arch):
        # Construct the network (Feedfoward deterministic network)
        layers = []
        for ii in range(len(net_arch)):
            layers.append(nn.Linear(net_arch[ii][0], net_arch[ii][1]))
            if ii < len(net_arch) - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.loss_fcn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
    
    def predict(self, obs):
        return self.model(torch.tensor(obs, dtype=torch.float32))
    
    def train(self, states, actions):
        print("Training:")
        # Format training and testing data, use 20% for validation
        # note actions / 2 since actions in [-2, 2] and final layer of output is tanh in [-1, 1]
        X_train, X_test, y_train, y_test = train_test_split(states, actions / 2, train_size=0.8, shuffle=True)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        # training parameters
        n_epochs = 3   # number of epochs to run
        batch_size = 32  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
        
        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        
        # training loop
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")
            self.model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = self.model(X_batch)
                    loss = self.loss_fcn(y_pred, y_batch)
                    # backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    self.optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            self.model.eval()
            y_pred = self.model(X_test)
            mse = self.loss_fcn(y_pred, y_test)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(self.model.state_dict())
        
        # restore model and return best accuracy
        self.model.load_state_dict(best_weights)
        print(f"Loss: {best_mse}")