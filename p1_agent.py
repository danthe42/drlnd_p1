import numpy as np
import random
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4        # how often to update the network

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, memory, batch_size, LR=5e-4, GAMMA=0.95):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.GAMMA=GAMMA

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = memory

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                self.learn(self.memory.sample(device))

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # Use Double DQN algorithm here: 
        # Evaluate with the target actions with the online network, and calculate their Q values with the target network.  
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)    

        # Compute Q targets for current states 
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()

        # Compute loss
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class PRIOAgent(Agent):
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        idxs, states, actions, rewards, next_states, dones, weights = experiences

        # Use Double DQN algorithm here: 
        # Evaluate with the target actions with the online network, and calculate their Q values with the target network.  
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)    

        # Compute Q targets for current states 
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        deltas = Q_targets.detach().cpu().numpy() - Q_expected.detach().cpu().numpy()
        self.memory.batch_update( idxs, abs(deltas) )

        # Minimize the loss
        self.optimizer.zero_grad()
        
        # Compute loss
        
        loss = (weights * F.mse_loss(Q_expected, Q_targets)).mean()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)           
