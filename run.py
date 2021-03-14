from unityagents import UnityEnvironment
import numpy as np
from p1_agent import Agent, PRIOAgent
from collections import deque
import torch
from ReplayBuffer import ReplayBuffer, PrioReplayBuffer

seed = 0

env = UnityEnvironment(file_name="Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)
use_prioritized_experience_replay = True

if use_prioritized_experience_replay==True:
    
    BUFFER_SIZE = int(2e5)  # replay buffer size
    BATCH_SIZE = 32         # minibatch size
    EPS_START = 0.1
    EPS_END = 0.
    EPS_DECAY = 0.98
    ALPHA = 0.6             # [0~1] convert the importance of TD error to priority
    BETA = 0.6              # importance-sampling, from initial value increasing to 1
    BETA_INC_PER_SAMPLING = 0.000003
    GAMMA = 0.98            # discount factor       # pipa
    LR = 5e-4               # learning rate 

    memory = PrioReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, ALPHA, BETA, BETA_INC_PER_SAMPLING)
    agent = PRIOAgent(state_size=state_size, action_size=action_size, seed=seed, memory=memory, batch_size=BATCH_SIZE, LR=LR, GAMMA=GAMMA)
    netname = "net_prioddqn.pth"

else:
    
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    EPS_START = 0.1
    EPS_END = 0.
    EPS_DECAY = 0.98
    GAMMA = 0.99            # discount factor
    LR = 5e-4               # learning rate 

    memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    agent = Agent(state_size=state_size, action_size=action_size, seed=seed, memory=memory, batch_size=BATCH_SIZE, LR=LR, GAMMA=GAMMA)
    netname = "net_ddqn.pth"

try:
    agent.qnetwork_local.load_state_dict(torch.load(netname))
    print("Network loaded from file: ", netname)
except FileNotFoundError:
    pass

def dqn(n_episodes=5000, max_t=1000, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0   
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), netname)
            break
    return scores

scores = dqn()    

print("\nScores: {}".format(scores))

env.close()

