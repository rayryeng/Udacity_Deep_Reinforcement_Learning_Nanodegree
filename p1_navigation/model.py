import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=37, action_size=4, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Note - the state vector are simply features so we can just use
        # fully connected layers
        # For the banana learning task, we have 37 features coming in, 4 features coming out
        self.fc1 = nn.Linear(state_size, 150, bias=False)  # BN layers already have a learnable bias
        self.bn1 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 120, bias=False)
        self.bn2 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.relu(self.bn1(self.fc1(state)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class QNetworkConvolutional(nn.Module):
    """Actor (Policy) Model using Convolutional Layers."""

    def __init__(self, input_size=(3, 84, 84), action_size=4, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            input_size (tuple): Size of an input image
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkConvolutional, self).__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Original DQN paper uses three conv layers + 2 FC layers
        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Number of neurons after flattening is 5 x 5 x 128
        self.num_neurons_flatten = ((input_size[1] // 16) * (input_size[2] // 16) * 128)

        self.fc1 = nn.Linear(self.num_neurons_flatten, 512, bias=False)  # BN layers already have a learnable bias
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, action_size)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Conv-BN-ReLU-Pool #1
        # Conv with stride 2 brings it down to 42 x 42
        # Pool goes down to 21 x 21
        x = self.pool(self.relu(self.bn1(self.conv1(state))))

        # Conv-BN-ReLU-Pool #2
        # Pool brings it down to 10 x 10
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Conv-BN-ReLU-Pool #3
        # Pool brings it down to 5 x 5
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, self.num_neurons_flatten)

        # FC-BN-ReLU #1
        x = self.relu(self.bn4(self.fc1(x)))

        # FC
        x = self.fc2(x)

        return x
