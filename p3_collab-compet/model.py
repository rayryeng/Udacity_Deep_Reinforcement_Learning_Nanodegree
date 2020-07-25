import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    Given a PyTorch layer, return the lower and upper bounds of the
    uniform distribution limits to initialise the layer
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units, bias=False)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=False)
        self.fc3 = nn.Linear(fc2_units, action_size)

        ## Added by Ray - Added batch norm for quicker training
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialise the neural network weights upon creating the Actor
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        ## Added by Ray - Added batch norm for quicker training
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        # Changed from F.tanh to torch.tanh due to deprecation warning
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units, bias=False)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units, bias=False)
        self.fc3 = nn.Linear(fc2_units, 1)

        ## Added by Ray - Added batch norm for quicker training
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialise the neural network weights upon creating the Actor
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        ## Added by Ray - Added batch norm for quicker training
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)