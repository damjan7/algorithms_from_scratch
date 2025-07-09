import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import torch
import torch.nn as nn
import torch.functional as F

# create expert and corresponding expert rollouts for imitation learning algorithms
env = gym.make("CartPole-v1")
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)
expert.learn(100000)  # Note: set to 100000 to train a proficient expert

from stable_baselines3.common.evaluation import evaluate_policy

reward, _ = evaluate_policy(expert, env, 10)
print(reward)

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)

print(
    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
"""
)

## done

""""
BEHAVIORAL CLONING

transitions contains all the states and actions, can use to train behavioral cloning
state is a continuous 4-tuple, actions is binary
"""

for i, x in enumerate(transitions):
    print(x.keys())
    break  

# define net structure for BC as benachmark

# simple MLP for behavioral cloning
import torch
import torch.nn as nn
import torch.functional as F

class BC_Net_Cartpole(nn.Module):
    def __init__(self, hidden_size, input_size=4, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x).flatten()

  states = np.array([x['obs'] for x in transitions])
actions = np.array([x['acts'] for x in transitions])

from torch.utils.data import Dataset
class ExpertDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
    
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

dataset = ExpertDataset(X=states, y=actions)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# train loop

model = BC_Net_Cartpole(hidden_size=16)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        # Backward pass and optimization
        optimizer.zero_grad()  
        loss.backward()         
        optimizer.step()        
        if batch_idx % 500 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

model.eval()

# also "train" an undertrained model to compare
undertrained_model = BC_Net_Cartpole(hidden_size=16)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(undertrained_model.parameters(), lr=0.0001)

num_epochs = 2

for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        outputs = undertrained_model(inputs)
        loss = loss_fn(outputs, targets)
        # Backward pass and optimization
        optimizer.zero_grad()  
        loss.backward()         
        optimizer.step()        
        if batch_idx % 500 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

undertrained_model.eval()

"""
Since we're working with the imitation package for some funcs,
define Custom Policy Class that takes our policy NN as a parameter to predict
Note; here the output of the net is transformed to Long, since I want deterministic actions
"""

from typing import Dict, Tuple
from numpy import ndarray
from stable_baselines3.common.policies import BasePolicy

# define custom policy class that takes a net as a param to predict
# is basically actor critic without the critic!?
class CustomPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, behavioral_cloning_net):
        super().__init__(observation_space=observation_space, action_space=action_space)
        self.net = behavioral_cloning_net
        self.action_space = action_space

    def _predict(self, 
                 observation,
                 state=None,
                 episode_start=None,
                 deterministic=True,):
        return self.net.forward(observation).round().long()

model.eval()
base_BC_policy = CustomPolicy(env.observation_space, env.action_space, model)

reward, _ = evaluate_policy(base_BC_policy, env, 10)
print(reward)  

undertrained_model.eval()
untrained_base_BC_policy = CustomPolicy(env.observation_space, env.action_space, undertrained_model)

reward, _ = evaluate_policy(untrained_base_BC_policy, env, 10)
print(reward)  


# render model
from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = DummyVecEnv([lambda: env])
obs = vec_env.reset()
for i in range(400):
    action, _states = base_BC_policy.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
