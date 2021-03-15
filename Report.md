

[TOC]



# Project 1: Navigation - Report

2 different agents were created in this project to see their efficiency in solving the problem:

- One, using the Double Q-learning algorithm (see 2nd paper in the References section for details),
- And an other one, similar to the previous one, but it's replay buffer is not random. It follows the algorithm described in the 3rd referenced document. 



## Architecture

I have create a hybrid solution in python using the numpy and pytorch frameworks for the agents, and using a UnityEnironment from the unityagents library as the bridge between the simulated environment implemented in Unity. 

The agents can be trained in console mode, or using a Jupyter notebook. The files:

- model.py: The neural network implementing the policy model of the agents with 3 FC layers and 2 ReLUs.

- SumTree.py: A helper object implementing a binary tree where each node's value is the sum of it's children's. It's necessary for the implementation of the prioritized experience replay. (see 4th link in References for the source)

- ReplayBuffer.py: Implementation of buffers to be used by the agents for storing the transitions and sampling from them. This file contains 2 such buffers: ReplayBuffer and PrioReplayBuffer class. The first one is a simple transition buffer with completely random sampling. The latter implements the prioritized replay buffer, with the help of the previous, SumTree class.

-  p1_agent.py: Here are the agents: The "Agent" class implements the Double DQN algorithm, and the "PRIOAgent" class implements the Double DQN with Prioritized Experience Replay agent. The class extends the previous one, and enhances it further by using a modified learn() step which updates the priorities in the PrioReplayBuffer class. 

- run.py: This file can be used to train an agent from console by directly executing it. There is a constant at the beginning (use_prioritized_experience_replay) which can be used to select which agent to train.

- Navigation.ipynb: A Jupyter notebook for demonstrating 

  - the train process of both agents, 
  - the results are visualized,
  - after training the models are loaded and inference mode is also demonstrated on both agents.   

  This file is also exported here, so you can read it without an installed Jupyter notebook instance: [exported notebook](export/Navigation.md)



Please follow the previous notebook to understand the training of the two Banana Collector agents.



## The Agents

These agents, the hyperparameters, the roles, the different methods are explained in the code comments, so I will only give a summarized overview about the agents here.   



### Double DQN Agent

The normal Double DQN (DDQN) agent realizes the ideas described in the 2nd referenced material: 

- It follows epsilon-greedy policy to determine the next action.
- It uses two neural networks. The evaluation of which actions to follow, and the estimation of the values of these actions are separated, they are occuring on these two networks: One on the on-line network which is used to learn (backprop.), and the other on a "freezed" target network which is only updated at the end of the learning step. This helps to minimize the overoptimism experiences in standard DQNs.
-  This agent is using the fixed-sized randomly sampling experience replay buffer from which it selects transitions using the uniform distribution.
- The chosen Hyperparameter values for this agent:
  - BUFFER_SIZE = int(1e5)  # replay buffer size
  - BATCH_SIZE = 64         # minibatch size
  - EPS_START = 0.1          # Starting Epsilon to be used for the Epsilon Greedy algorithm
  - EPS_END = 0.               # Ending Epsilon to be used for the Epsilon Greedy algorithm
  - EPS_DECAY = 0.98       # Epsilon Decay to be used for the Epsilon Greedy algorithm
  - GAMMA = 0.99            # discount factor for calculation of future rewards
  - LR = 5e-4                      # learning rate 
  - Policy model can also be considered a hyperparameter: neural network with 3 Fully Connected layers, with 2 ReLU activation functions between them. The layer sizes are: [ 256, 128, 4 ].



### Double DQN Agent + Prioritized Experience Replay

The second agent is based on the previous one, and enhances it with an additional feature: the Prioritized Experience Replay. It is described in the 3rd referenced document in detail. It's main points:

- This agent is using the PrioReplayBuffer class for its transition memory buffer. This special buffer uses the "proportional prioritization" variant for sampling. 
- Alpha, Beta, Beta_Decay, and a small positive Epsilon value (to avoid zero probabilities) are four additional hyperparameters which are used in this special ReplayBuffer class. 
- The chosen Hyperparameter values for this agent:
  - BUFFER_SIZE = int(2e5)  # replay buffer size
  - BATCH_SIZE = 32         # minibatch size
  - EPS_START = 0.1          # Starting Epsilon to be used for the Epsilon Greedy algorithm
  - EPS_END = 0.               # Ending Epsilon to be used for the Epsilon Greedy algorithm
  - EPS_DECAY = 0.98       # Epsilon Decay to be used for the Epsilon Greedy algorithm
  - ALPHA = 0.6                 # [0~1] convert the importance of TD error to priority
  - BETA = 0.6                    # importance-sampling, from initial value increasing to 1
  - BETA_INC_PER_SAMPLING = 0.000003     # importance-sampling inc./sample
  - GAMMA = 0.98            # discount factor for calculation of future rewards
  - LR = 5e-4                      # learning rate 
  - Policy model can also be considered a hyperparameter: neural network with 3 Fully Connected layers, with 2 ReLU activation functions between them. The layer sizes are: [ 256, 128, 4 ].



### Results, Conclusion

After training both agents and comparing them, I got two important results:

- Prioritized Replay solved the task in fewer episodes, which means it's learning is more efficient than the standard Double DQN's.
- After "Banana Collecting" on a few episodes, I've noticed much smaller variations among the results of the 2nd agent. This means that the general quality of the policy is much higher, this agent generalizes better.   

The task can be solved in relatively short time of training using Deep Q Networks, by trying only two possible enhancements on the original technic.  

This DQN learning seems to be very promising solving problems where the reward only comes up sometime later, in the future.



## Ideas for future work

After this implementation, and comparision of these two agents' performance the following possible ideas and promising directions would be worthwhile to work on: 

- Rank-based prioritization would be an easy to do modification (on the DDQN+Prioritized Exp. Replay agent). It would be a good idea to implement a 3rd agent with this algo. 
- Check out using Dropout layers, maybe it will help in generalization, and would decrease overlearning.
- Hyperparameter tuning: There's never enough CPU time for training... 
- Cuda, just an idea: Can we upload all transitions in the Experience Replay Buffer to GPU ? Could it be more efficient performance-wise ?        
- Other DQN enhancements should be incorporated. For example: Rainbow, Dueling network architectures, Asynchronous Methods, Distributional Perspective  
- Learn from pixels: Solve the same problem with the following modification: inputs are raw pixel images instead of the current state space of 37 Real numbers. 



## PS 

Originally, I wanted to attach a video about the Banana Collection to this report. I've tried to use the Unity simulator on 2 PCs with Win64, and 1 PC with Ubuntu Linux OSs installed, without luck. It's possible that the Unity executable is too old, or my GPU cards are too new. Fortunately I could use the headless Simulator in my local Ubuntu Linux, and on the Workspace to train my agents.

As it's not a requirement in the project rubric, we can ignore it for now.



## REFERENCES

Many Google DeepMind researchers - Human-level control through deep reinforcement learning - https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Hado van Hasselt and Arthur Guez and David Silver (Google DeepMind) - *Deep Reinforcement Learning with Double Q-learning* https://arxiv.org/abs/1509.0646 

TomSchaul, John Quan, Ioannis Antonoglou and David Silver (Google Deepmind) - *Prioritized Experience Replay* https://arxiv.org/abs/1511.05952 

SumTree python implementation: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py (MIT License)











