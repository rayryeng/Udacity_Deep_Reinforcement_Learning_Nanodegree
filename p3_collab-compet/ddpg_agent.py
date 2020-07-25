import numpy as np
import random
import copy
from collections import namedtuple, deque
from pprint import pprint

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

### Default parameters - these can be changed when you specify an input
### dictionary when creating the agents
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.       # L2 weight decay
FC1_UNITS = 400         # Number of hidden units for the first hidden layer of the Actor and Critic networks
FC2_UNITS = 300         # Number of hidden units fro the second hidden layer of the Actor and Critic networks

## Added by Ray
UPDATE_EVERY = 20        # At every multiple of this value, we update our actor and critic
NUM_ITERS_LEARN = 10     # When we finally do the update, we run the learning process this many times

config_default = {
    'batch_size': BATCH_SIZE,
    'buffer_size': BUFFER_SIZE,
    'gamma': GAMMA,
    'tau': TAU,
    'lr_actor': LR_ACTOR,
    'lr_critic': LR_CRITIC,
    'weight_decay': WEIGHT_DECAY,
    'update_every': UPDATE_EVERY,
    'num_iters_learn': NUM_ITERS_LEARN,
    'fc1_units': FC1_UNITS,
    'fc2_units': FC2_UNITS
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agents():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed, config: dict = {}):
        """Initialize an Agents object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
            config (dict): dictionary of parameters to specify for the agents if
                           one decides to not use the default parameters.  Specify
                           an empty dictionary to just use the default params. The
                           available keys and their defaults are:
                           - buffer_size - replay buffer size: 1000000
                           - batch_size - mini batch size: 64
                           - gamma - discount factor: 0.99
                           - tau - soft update for parameters: 0.001
                           - lr_actor - learning rate of the actor - 0.0001
                           - lr_critic - learning rate of the critic - 0.001
                           - weight_decay - L2 weight decay for the model - 0
                           - fc1_units - number of first hidden layer units - 400
                           - fc2_units - number of second hidden layer units - 300
                           - update_every - interval to perform soft updates - 20
                           - num_iters_learn - number of iteration steps at an update - 10
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents  # Added by Ray - store the number of agents
        random.seed(random_seed)
        self.num_timesteps = 0  # Added by Ray - store the number of timesteps every time we step
        self.config = config_default.copy()

        if len(config) != 0:
            print('Using user-defined parameters')
            for k in config:
                if k.lower() in config:
                    self.config[k.lower()] = config[k.lower()]
        else:
            print('Using default hyperparameters')

        pprint(self.config)

        # Actor Network (w/ Target Network)
        ## Note - Each agent's states and actions come from Unity as independent observations of the same
        ## environment so we can simply use one Actor and Critic and the batch size is simply the number
        ## of agents in play
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=self.config['fc1_units'], fc2_units=self.config['fc2_units']).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=self.config['fc1_units'], fc2_units=self.config['fc2_units']).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config['lr_actor'])

        # Critic Network (w/ Target Network)
        ## Note - Each agent's states and actions come from Unity as independent observations of the same
        ## environment so we can simply use one Actor and Critic and the batch size is simply the number
        self.critic_local = Critic(state_size, action_size, random_seed, fcs1_units=self.config['fc1_units'], fc2_units=self.config['fc2_units']).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fcs1_units=self.config['fc1_units'], fc2_units=self.config['fc2_units']).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config['lr_critic'], weight_decay=self.config['weight_decay'])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory - note: all agents share this memory
        self.memory = ReplayBuffer(self.config['buffer_size'], self.config['batch_size'], random_seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experiences in replay memory, and use random sample from buffer to learn."""

        ## Added by Ray - Sanity checks to make sure the incoming lengths and shapes make sense
        assert(states.shape[0] == self.num_agents)
        assert(states.shape[1] == self.state_size)
        assert(next_states.shape[0] == self.num_agents)
        assert(next_states.shape[1] == self.state_size)
        assert(actions.shape[0] == self.num_agents)
        assert(actions.shape[1] == self.action_size)
        assert(len(rewards) == self.num_agents)
        assert(len(dones) == self.num_agents)

        ## Modified by Ray
        # Save experiences / rewards - iterate through each SARSD tuple from each agent, add it into
        # the memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Increment counter and reset if necessary
        self.num_timesteps = (self.num_timesteps + 1) % self.config['update_every']

        # Learn, if enough samples are available in memory
        if len(self.memory) >= self.config['batch_size']:
            ## Added by Ray -
            # If we reach the number of timesteps needed to update...
            if self.num_timesteps == 0:
                # Update the parameters for NUM_ITERS_LEARN times
                for _ in range(self.config['num_iters_learn']):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.config['gamma'])
        
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        ## Added by Ray - Sanity checks to make sure the incoming lengths and shapes make sense
        assert(states.shape[0] == self.num_agents)
        assert(states.shape[1] == self.state_size)

        # Convert input states into Tensor
        states = torch.from_numpy(states).float().to(device)

        # Set Actor network to eval mode due to Batch Norm
        self.actor_local.eval()

        # Now collect the actions
        with torch.no_grad():
            ## Output should be num_agents x num_actions
            actions = self.actor_local(states).cpu().data.numpy()
        
        # Now set to train mode again for training
        self.actor_local.train()

        ## Modified by Ray - Iterate through each action and add noise to it
        if add_noise:
            final_actions = [action + self.noise.sample() for action in actions]
        else:
            # Just keep a view of it if we haven't added any noise
            final_actions = actions

        # Stack the NumPy arrays together to generate a num_agents x num_actions array
        final_actions = np.vstack(actions)
        return np.clip(final_actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        ## Added by Ray - Perform gradient clipping for more stability
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config['tau'])
        self.soft_update(self.actor_local, self.actor_target, self.config['tau'])                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        ## Fixed bug from original Udacity implementation where the normal
        ## distribution needs to be used instead of a uniform distribution
        dx = self.theta + (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
