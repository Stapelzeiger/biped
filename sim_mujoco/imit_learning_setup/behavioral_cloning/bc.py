import copy
import numpy as np

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

"""Behavioral Cloning
This class implements behavioral cloning in the most naive way. 
Expert rollouts are passed to the algorithm, as well as a policy network.
The policy network is trained directly on the expert rollouts to predict the
action taken by the expert.
"""
class bc:
    
    def __init__(self, states, actions, policy_arch, batch_size=32, lr=0.0001, val_split=0.2, logging_path=None):
        """Initializes a behavioral cloning algorithm

        Args:
            states (array-like): Set of states visited by the expert
            actions (array-like): Set of actions taken by expert in 'states'
            policy_arch (list-like): Network architecture for behavioral cloning policy
                Elements are dictionaries containing keys 'Layer' ('Linear', 'ReLU', 'Tanh', ...)
                and 'Input' and 'Output' dimension where relevant.
            batch_size (int, optional): Batch size for learning. Defaults to 32.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.0001.
            val_split (float, optional): Percentage of data to use for validation. Defaults to 0.2.
            logging_path (_type_, optional): Path to save learning results to. Defaults to None.
        """
        # Check to make sure dimensions of everything lines up
        assert policy_arch[0]['Input'] == states.shape[1], "Network input does not match size of states"
        assert policy_arch[-1]['Output'] == actions.shape[1], "Network output does not match size of actions"

        # Initialize class variables
        self.exp_states = states
        self.exp_actions = actions
        mean_state = np.mean(states, 0)
        mean_action = np.mean(actions, 0)
        std_state = np.std(states, 0)
        std_action = np.std(actions, 0)
        # Pass mean and std of states, actions to the agent, so that the agent can
        # normalize inputs/outputs of the neural network
        self.bc_agent = BC_Agent(policy_arch, mean_state, std_state, mean_action, std_action)
        self.logging_path = logging_path
        self.loss_fcn = nn.MSELoss()
        self.optimizer = optim.Adam(self.bc_agent.parameters(), lr=lr, eps=1e-5)
        self.batch_size = batch_size

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(self.exp_states, self.exp_actions, train_size=1-val_split, shuffle=True)
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

        # Get the device for training
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.set_device()

    def train(self, epochs, use_best_weights=True):
        """Trains the policy for a set number of epochs. 

        Args:
            epochs (int): number of epochs to train
            use_best_weights (bool, optional): Whether to use validation loss to save best iteration. Defaults to True
        """
        writer = SummaryWriter()
        
        # training parameters
        batch_start = torch.arange(0, len(self.X_train), self.batch_size)
        
        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        
        # training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self.bc_agent.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = self.X_train[start:start+self.batch_size].to(self.device)
                    y_batch = self.y_train[start:start+self.batch_size].to(self.device)
                    # forward pass
                    y_pred = self.bc_agent.get_action(X_batch)
                    loss = self.loss_fcn(y_pred, y_batch)
                    # backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    self.optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            self.bc_agent.eval()
            y_pred = self.bc_agent.get_action(self.X_test.to(self.device))
            mse = self.loss_fcn(y_pred, self.y_test.to(self.device))
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(self.bc_agent.state_dict())
            writer.add_scalar('BC_Loss/eval_MSE', mse, epoch)
            print(f"\tLoss: {mse}")
        # restore model and return best accuracy
        if use_best_weights:
            self.bc_agent.load_state_dict(best_weights)
            print(f"Final Loss: {best_mse}")
        else:
            print(f"Final Loss: {mse}")

    def save_policy(self, path):
        """Saves the current policy weights to a designated path

        Args:
            path (path-like): location to save the current policy weights
        """
        self.bc_agent.save_policy(path)

    def load_policy(self, path):
        """Loads policy weights from the designated path

        Args:
            path (path-like): location to read policy weights
        """
        self.bc_agent.load_policy(path)

    def set_states_actions(self, states, actions):
        """Sets the set of states and actions taken by the expert

        Args:
            states (_type_): Set of states visited by the expert
            actions (_type_): Set of actions taken by expert in 'states'
        """
        self.exp_states = states
        self.exp_actions = actions

    def set_device(self):
        """Sets the device to be used for training
        """
        # device = (
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if torch.backends.mps.is_available()
        #     else "cpu"
        # )
        self.bc_agent.to(self.device)
    

class BC_Agent(nn.Module):

    def __init__(self, policy_arch, exp_state_mean, exp_state_std, exp_action_mean, exp_action_std):
        """Initialize a Behavioral Cloning Agent

        Args:
            policy_arch (iterable): Network architecture for behavioral cloning policy
                Elements are dictionaries containing keys 'Layer' ('Linear', 'ReLU', 'Tanh', ...)
                and 'Input' and 'Output' dimension where relevant.
            exp_state_mean (array-like): Means of expert states in training data for normalization
            exp_state_std (_type_): Standard Deviations of expert states in training data for normalization
            exp_action_mean (_type_): Means of expert actions in training data for normalization
            exp_action_std (_type_): Standard Deviations of expert actions in training data for normalization
        """
        super(BC_Agent, self).__init__()
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.policy = BC_Agent._gen_policy(policy_arch)

        # Include mean/std of states/actions as parameters in the model
        # requires_grad=False will prevent these parameters from being changed during training
        # this way when model is saved, means/stds will be saved, and likewise they will be loaded 
        self.state_mean = nn.Parameter(torch.Tensor(exp_state_mean).to(self.device), requires_grad=False)
        self.state_std = nn.Parameter(torch.Tensor(exp_state_std).to(self.device), requires_grad=False)
        self.action_mean = nn.Parameter(torch.Tensor(exp_action_mean).to(self.device), requires_grad=False)
        self.action_std = nn.Parameter(torch.Tensor(exp_action_std).to(self.device), requires_grad=False)

    staticmethod
    def _gen_policy(policy_arch):
        """Generates a torch.nn Sequential model from the policy architecture

        Args:
            policy_architecture (iterable): Iterable definiting policy architecture
                with 'Layer', 'Input', 'Output', and other arguments for each layer
        """
        layers = []
        # Loop through layers
        for layer in policy_arch:
            if layer['Layer'] == 'Linear':
                if 'SpectralNorm' in layer.keys() and layer['SpectralNorm']:
                    # Add a linear layer with spectral normalization
                    layers.append(nn.utils.spectral_norm(nn.Linear(layer['Input'], layer['Output'])))
                else:
                    # Add a linear layer without spectral normalization
                    layers.append(nn.Linear(layer['Input'], layer['Output']))
            elif layer['Layer'] == 'ReLU':
                # Add a ReLU layer
                layers.append(nn.ReLU())
            elif layer['Layer'] == 'Tanh':
                # Add a Tanh layer
                layers.append(nn.Tanh())
            else:
                ValueError(f"Layer type not recognized: {layer}")
        # And create the model by unpacking the list of layers
        model = nn.Sequential(*layers)
        return model
    
    def get_action(self, x):
        """Return the action taken by the agent in state x

        Args:
            x (array-like): The state(s) to query action(s) for.

        Returns:
            array-like: actions(s) to be taken in the queried state(s)
        """
        # print("self.device: ", self.device)
        return self.policy((x.to(self.device) - self.state_mean) / self.state_std) * self.action_std + self.action_mean

    def save_policy(self, path):
        """Saves the current policy weights to a designated path

        Args:
            path (path-like): location to save the current policy weights
        """
        torch.save(self.state_dict(), path)

    def load_policy(self, path):
        """Loads policy weights from the designated path

        Args:
            path (path-like): location to read policy weights
        """
        self.load_state_dict(torch.load(path))
