import torch.optim as optim

from model import *
from .train import *


def init_ippo(obs_dim, action_dim, num_agents, device):
    policies = [PPONet(obs_dim, action_dim).to(device) for _ in range(num_agents)]
    optimizers = [optim.Adam(p.parameters(), lr=3e-4) for p in policies]
    return [PPOTrainer(p, o) for p, o in zip(policies, optimizers)]


def actor_ippo(trainers, i, obs, device):
    action, log_prob, _ = trainers[i].policy.act(torch.tensor(obs[i], dtype=torch.float32).to(device))
    return action, log_prob, _


def train_ippo(env, device, num_agents, num_epochs):
    train(env, device, init_ippo, actor_ippo, num_agents, num_epochs)