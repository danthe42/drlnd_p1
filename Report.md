[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"



Table of Contents:

[TOC]



# Project 1: Navigation - Report

2 different agents were created in this project to see their efficiency in solving the problem:

- One, using the Double Q-learning algorithm (see 2nd paper in the References section for details),
- And an other one, similar to the previous one, but it's replay buffer is not random. It follows the algorithm described in the 3rd referenced document. 

### Architecture

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

  This file is also exported here: [exported notebook](export/Navigation.md)



For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.



REFERENCES

Many Google DeepMind researchers - Human-level control through deep reinforcement learning - https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Hado van Hasselt and Arthur Guez and David Silver (Google DeepMind) - *Deep Reinforcement Learning with Double Q-learning* https://arxiv.org/abs/1509.0646 

TomSchaul, John Quan, Ioannis Antonoglou and David Silver (Google Deepmind) - *Prioritized Experience Replay* https://arxiv.org/abs/1511.05952 

SumTree python implementation: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py (MIT License)











