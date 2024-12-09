import torch
import torch.nn as nn


# PPO策略网络定义
class PPONet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PPONet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        action_probs = self.actor(obs)
        value = self.critic(obs)
        return action_probs, value

    def act(self, obs):
        action_probs = self.actor(obs)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs, actions):
        action_probs = self.actor(obs)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(obs)
        return log_probs, entropy, value