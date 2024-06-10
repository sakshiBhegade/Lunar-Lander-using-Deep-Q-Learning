# Lunar Lander using Deep Q-Learning

This repository contains a Deep Q-Learning implementation for the Lunar Lander environment from the Gymnasium library. This project implements a Deep Q-Learning algorithm for training an agent to navigate and land a lunar module safely on the moon's surface. Key components of the project include a neural network architecture for approximating the action-value function, experience replay for stabilizing training, and the Deep Q-Learning algorithm for iterative learning. The agent learns to optimize its actions based on rewards received from the environment, ultimately achieving the goal of landing the lunar module safely and efficiently.


https://github.com/sakshiBhegade/Lunar-Lander-using-Deep-Q-Learning/assets/144518865/7355e52d-29cd-4a9a-912b-be939ed8e134


## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Algorithm Details](#algorithm-details)
4. [Implementation](#implementation)
5. [Training the Agent](#training-the-agent)
6. [Visualizing Results](#visualizing-results)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Deep Q-Learning is an extension of Q-Learning, where a neural network is used to approximate the Q-value function. This project uses the Lunar Lander environment from the Gymnasium library to demonstrate the application of Deep Q-Learning.

## Installation

To set up the environment for this project, follow these steps:

1. **Install Gymnasium:**
    ```bash
    pip install gymnasium
    ```

2. **Install Atari environments:**
    ```bash
    pip install "gymnasium[atari, accept-rom-license]"
    ```

3. **Install SWIG (Simplified Wrapper and Interface Generator):**
    ```bash
    apt-get install -y swig
    ```

4. **Install Box2D environments:**
    ```bash
    pip install gymnasium[box2d]
    ```

## Algorithm Details

### Neural Network Architecture

The neural network used in this project consists of three fully connected layers:

Input Layer:
Input: State vector 
𝑥
∈
𝑅
state_size
x∈R 
state_size
 
Example: 
𝑥
=
[
𝑥
1
,
𝑥
2
,
𝑥
3
,
𝑥
4
]
x=[x 
1
​
 ,x 
2
​
 ,x 
3
​
 ,x 
4
​
 ]
First Fully Connected Layer:
Weights: 
𝑊
1
∈
𝑅
state_size
×
64
W 
1
​
 ∈R 
state_size×64
 
Biases: 
𝑏
1
∈
𝑅
64
b 
1
​
 ∈R 
64
 
Output: 
ℎ
1
=
ReLU
(
𝑊
1
𝑥
+
𝑏
1
)
h 
1
​
 =ReLU(W 
1
​
 x+b 
1
​
 )
Dimension: 
ℎ
1
∈
𝑅
64
h 
1
​
 ∈R 
64
 
Second Fully Connected Layer:
Weights: 
𝑊
2
∈
𝑅
64
×
64
W 
2
​
 ∈R 
64×64
 
Biases: 
𝑏
2
∈
𝑅
64
b 
2
​
 ∈R 
64
 
Output: 
ℎ
2
=
ReLU
(
𝑊
2
ℎ
1
+
𝑏
2
)
h 
2
​
 =ReLU(W 
2
​
 h 
1
​
 +b 
2
​
 )
Dimension: 
ℎ
2
∈
𝑅
64
h 
2
​
 ∈R 
64
 
Output Layer:
Weights: 
𝑊
3
∈
𝑅
64
×
action_size
W 
3
​
 ∈R 
64×action_size
 
Biases: 
𝑏
3
∈
𝑅
action_size
b 
3
​
 ∈R 
action_size
 
Output: 
𝑦
=
𝑊
3
ℎ
2
+
𝑏
3
y=W 
3
​
 h 
2
​
 +b 
3
​
 
Dimension: 
𝑦
∈
𝑅
action_size
y∈R 
action_size
 
# Experience Replay

Experience replay is used to stabilize training by reusing past experiences.

# Deep Q-Learning Algorithm
<img width="744" alt="deep_learning_lunar" src="https://github.com/sakshiBhegade/Lunar-Lander-using-Deep-Q-Learning/assets/144518865/ae459d2c-cd49-4535-a2a5-b08c4a1ef6be">

## Implementation

Here is the detailed implementation of the Deep Q-Learning algorithm:

### Importing Libraries

```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
```

### Neural Network Definition

```python
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### Experience Replay Class

```python
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))
    
    def __len__(self):
        return len(self.memory)
```

### DQN Agent Class

```python
class Agent:
    def __init__(self, state_size, action_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.qnetwork_local = Network(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = Network(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)
        
        self.memory = ReplayMemory(int(1e5))
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3
        self.update_every = 4
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
    
    def act(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
```

## Training the Agent

```python
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent

 = Agent(state_size, action_size, seed=42)

def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window)>=200.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores
```

## Visualizing Results

After training, you can visualize the results by plotting the scores:

```python
import matplotlib.pyplot as plt

scores = train()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```
