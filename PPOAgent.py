import torch
import random
from torch import nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.forward(state)
            action = torch.multinomial(action_probs, num_samples=1)

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, action_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []


class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, epsilon=0.2, update_steps=10):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_steps = update_steps

    def get_action_probs(self, state):
        return self.actor(state)

    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = torch.cat((values, torch.tensor([0.0], dtype=torch.float32, device=values.device)))
        # print(len(rewards), len(values), len(dones))  # 4096, 4097, 4096
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).to(device)

    def update(self, memory):
        states = torch.stack(memory.states).to(device)
        actions = torch.tensor(memory.actions).to(device)
        old_probs = torch.tensor(memory.action_probs).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(memory.dones, dtype=torch.float32).to(device)

        # Compute discounted rewards and advantages
        values = self.critic(states).detach().squeeze()
        # print(values.shape)  # 4096

        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = self.compute_advantages(rewards, values, dones)

        # Update actor and critic
        for _ in range(self.update_steps):
            # Actor loss
            action_probs = self.actor(states)
            # print(action_probs.shape)  # torch.Size([4096, 9])
            new_probs = action_probs.gather(1, actions.unsqueeze(1).long()).squeeze()
            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = ((returns - values) ** 2).mean()

            # Backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        memory.clear()
