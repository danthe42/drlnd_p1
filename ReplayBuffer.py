from SumTree import SumTree
from collections import namedtuple, deque
import random
import torch
import numpy as np

"""
The ReplayBuffer class contains a collection of state transitions with rewards, so it's basically the "memory" of the agent.    
The sampling, which is recollecting transitions in a minibatch from this memory is completely random.
The maximum size of this replay buffer is fixed, defined in a hyperparameter. 
"""

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

"""
The PrioReplayBuffer class extends the standard ReplayBuffer class with the following function:
- The sampling will happen using the absolute error-based, proportional prioritization of transitions from the memory. 
   (see Report.md description for details)  
- It is using the SumTree class, which makes it possible to effectively sample from this distribution.  
"""
class PrioReplayBuffer:
 
    """Fixed-size buffer to store experience tuples."""
    eps = 0.01        # to avoid division by zero

    def __init__(self, action_size, buffer_size, batch_size, seed, _alpha=0., _beta=0., _beta_inc=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (int): [0~1] convert the importance of TD error to priority
            beta (int): importance-sampling, from initial value increasing to 1
            beta_inc (int): increment per sampling value
        """
        self.action_size = action_size
        self.buffer_size = buffer_size  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.tree = SumTree(buffer_size)
        self.length = 0
        self.alpha = _alpha
        self.beta = _beta
        self.beta_increment_per_sampling = _beta_inc

    # get max p value to use when adding a new entry to the memory 
    def get_max_p(self):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0.0:
            max_p = 1.0
        return max_p

    # add a new transition to memory
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.length = min(self.length+1, self.buffer_size)
        max_p = self.get_max_p();
        self.tree.add(max_p, (state, action, reward, next_state, done))                 # set the max p for new p

    # get a new minibatch of transitions, taken from the prioritized memory
    def sample(self, device):

        n = self.batch_size
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        _states = []
        _rewards = []
        _next_states = []
        _actions = []
        _dones = []
        pri_seg = self.tree.total_p / n                                                 # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])          # max = 1

        min_prob = self.eps + ( np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p )     # for later calculate ISweight

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            b_idx[i], p, data = self.tree.get_leaf(v)            
            _states.append( data[0] )
            _actions.append( data[1] )
            _rewards.append( data[2] )
            _next_states.append( data[3] )
            _dones.append( data[4] )
            prob = p / self.tree.total_p            
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)       # normalize weights, for stability reasons
        states = torch.FloatTensor( _states ).to(device)
        actions = torch.LongTensor( _actions ).unsqueeze(1).to(device)
        rewards = torch.FloatTensor( _rewards ).unsqueeze(1).to(device)
        next_states = torch.FloatTensor( _next_states ).to(device)
        dones = torch.FloatTensor( _dones ).unsqueeze(1).to(device)
        weights = torch.FloatTensor( ISWeights ).to(device)
        return ( b_idx, states, actions, rewards, next_states, dones, weights )

    # At the end of a learning step, we need to update the probabilities of the transitions in the given minibatch    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.eps              # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, 1.0)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)    
    
    def __len__(self):
        '''Return the current size of internal memory.'''
        return self.length
