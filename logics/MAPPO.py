import torch.optim as optim

from model import *
from .train import *


def init_mappo(obs_dim, action_dim, num_agents, device):
    policy = PPONet(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    return [PPOTrainer(policy, optimizer) for _ in range(num_agents)]


def actor_mappo(trainers, i, obs, device):
    action, log_prob, _ = trainers[i].policy.act(torch.tensor(obs[i], dtype=torch.float32).to(device))
    return action, log_prob, _


def train_mappo(env, device, num_agents, num_epochs):
    train(env, device, init_mappo, actor_mappo, num_agents, num_epochs)