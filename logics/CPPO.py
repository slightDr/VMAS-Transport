import torch.optim as optim

from model import *
from .train import *


def init_cppo(obs_dim, action_dim, num_agents, device):
    policy = PPONet(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    return [PPOTrainer(policy, optimizer)]


def actor_cppo(trainers, i, obs, device):
    action, log_prob, _ = trainers[0].policy.act(obs[i].clone().detach().to(device))
    return action, log_prob, _


def evaluator_cppo(trainers, i, obs, device):
    values = []
    for o in obs:
        values.append(
            trainers[0].policy.critic(torch.tensor(o, dtype=torch.float32).to(device)).item()
        )
    return values


def advantage_computer_cppo(trainers, i, rewards, values, dones):
    return trainers[0].compute_advantages(rewards, values, dones)


def trainer_cppo(trainers, i, trajectories):
    trainers[0].train(trajectories[i])


def train_cppo(env, device, num_epochs):
    train(env, device, init_cppo, actor_cppo, evaluator_cppo, advantage_computer_cppo, trainer_cppo, num_epochs)
