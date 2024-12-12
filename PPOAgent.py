import numpy as np
import torch
import random
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, state):
#         x = self.relu(self.fc1(state))
#         x = self.relu(self.fc2(x))
#         action_probs = self.softmax(self.fc3(x))
#         return action_probs


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        action_probs = self.softmax(self.fc2(x))
        return action_probs


# class Critic(nn.Module):
#     def __init__(self, state_dim, hidden_dim=256):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)
#         self.relu = nn.ReLU()
#
#     def forward(self, state):
#         x = self.relu(self.fc1(state))
#         x = self.relu(self.fc2(x))
#         value = self.fc3(x)
#         return value


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        value = self.fc2(x)
        return value


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def store(self, state, action, action_prob, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def sample(self, batch_size):
        """Randomly sample a batch of data from the memory."""
        if len(self.states) < batch_size:
            batch_size = len(self.states)

        indices = random.sample(range(len(self.states)), batch_size)
        sampled_states = [self.states[i] for i in indices]
        sampled_actions = [self.actions[i] for i in indices]
        sampled_action_probs = [self.action_probs[i] for i in indices]
        sampled_rewards = [self.rewards[i] for i in indices]
        sampled_dones = [self.dones[i] for i in indices]
        sampled_next_states = [self.next_states[i] for i in indices]

        return (
            torch.stack(sampled_states).to(device),
            torch.tensor(sampled_actions).to(device),
            torch.tensor(sampled_action_probs).to(device),
            torch.tensor(sampled_rewards).to(device),
            torch.tensor(sampled_dones).to(device),
            torch.stack(sampled_next_states).to(device),
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []


class PPOAgent:
    def __init__(self, actor, critic, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, lamb=0.95, epsilon=0.2, update_steps=4):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.update_steps = update_steps

    # def get_action_probs(self, state):
    #     return self.actor(state)

    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)

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
        old_log_probs = torch.tensor(memory.action_probs).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(memory.dones, dtype=torch.float32).to(device)
        next_states = torch.stack(memory.next_states).to(device)

        for _ in range(self.update_steps):
            # 价值网络
            next_state_value = self.critic(next_states)  # 下一时刻的state_value
            td_target = rewards + self.gamma * next_state_value * (1 - dones)  # 目标--当前时刻的state_value
            td_value = self.critic(states)  # 预测--当前时刻的state_value
            td_delta = td_value - td_target  # 时序差分  # [b,1]

            # 计算GAE优势函数，当前状态下某动作相对于平均的优势
            advantage = 0  # 累计一个序列上的优势函数
            advantage_list = []  # 存放每个时序的优势函数值
            td_delta = td_delta.cpu().detach().numpy()  # gpu-->numpy
            for delta in td_delta[::-1]:  # 逆序取出时序差分值
                advantage = self.gamma * self.lamb * advantage + delta
                advantage_list.append(advantage)  # 保存每个时刻的优势函数
            advantage_list.reverse()  # 正序
            advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(device)

            # 计算当前策略下状态s的行为概率 / 在之前策略下状态s的行为概率
            # print(self.actor(states).shape, actions.shape)  # torch.Size([500, 9]) torch.Size([500])
            log_probs = torch.log(self.actor(states).gather(1, actions.long().unsqueeze(-1)))
            ratio = log_probs / old_log_probs

            # clip截断
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage

            # 损失计算
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # clip截断
            critic_loss = torch.mean(nn.MSELoss()(td_value, td_target))  #
            # 梯度更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        memory.clear()

    # def update(self, memory):
    #     states = torch.stack(memory.states).to(device)
    #     actions = torch.tensor(memory.actions).to(device)
    #     old_probs = torch.tensor(memory.action_probs).to(device)
    #     rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
    #     dones = torch.tensor(memory.dones, dtype=torch.float32).to(device)
    #     next_states = torch.stack(memory.next_states).to(device)
    #
    #     # Compute discounted rewards and advantages
    #     values = self.critic(states).detach().squeeze()
    #     # print(values.shape)  # 4096
    #
    #     returns = []
    #     G = 0
    #     for reward, done in zip(reversed(rewards), reversed(dones)):
    #         G = reward + self.gamma * G * (1 - done)
    #         returns.insert(0, G)
    #     returns = torch.tensor(returns, dtype=torch.float32).to(device)
    #     advantages = self.compute_advantages(rewards, values, dones)
    #
    #     # Update actor and critic
    #     for _ in range(self.update_steps):
    #         # Actor loss
    #         action_probs = self.actor(states)
    #         # print(action_probs.shape)  # torch.Size([4096, 9])
    #         new_probs = action_probs.gather(1, actions.unsqueeze(1).long()).squeeze()
    #         ratio = new_probs / old_probs
    #         surr1 = ratio * advantages
    #         surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
    #         actor_loss = -torch.min(surr1, surr2).mean()
    #
    #         # Critic loss
    #         values = self.critic(states).squeeze()
    #         critic_loss = ((returns - values) ** 2).mean()
    #
    #         # Backpropagation
    #         self.actor_optimizer.zero_grad()
    #         actor_loss.backward()
    #         self.actor_optimizer.step()
    #
    #         self.critic_optimizer.zero_grad()
    #         critic_loss.backward()
    #         self.critic_optimizer.step()
    #
    #     memory.clear()

    def update_new(self, memory):
        # self.old_actor.load_state_dict(self.actor.state_dict())

        for step in range(self.update_steps):
            state, action, old_prob, reward, done, _ = memory.sample(64)
            # print(len(state), state[0].shape)  # (batch_size, action_dim)
            value = self.critic(state).squeeze()
            T = len(state)

            # advantages
            advantages = np.zeros(T, dtype=np.float32)
            for t in range(T):
                discount = 1
                a_t = 0
                for k in range(t, T - 1):
                    a_t += (reward[k] + self.gamma * value[k + 1] * (1 - done[k]) - value[k]) * discount
                    discount *= self.gamma * self.lamb
                advantages[t] = a_t
            with torch.no_grad():
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)

            # actor loss
            action_prob = self.actor(state)
            new_prob = action_prob.gather(1, action.unsqueeze(1).long()).squeeze()
            ratio = torch.exp(new_prob - old_prob)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()

            # critic loss
            ret = []
            G = 0
            for reward, done in zip(reversed(reward), reversed(done)):
                G = reward + self.gamma * G * (1 - done)
                ret.insert(0, G)
            ret = torch.tensor(ret, dtype=torch.float32).to(device)
            critic_loss = nn.MSELoss()(value, ret)

            # Backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def update_new_new(self, memory):
        for step in range(self.update_steps):
            old_state, action, old_prob, reward, done, new_state = memory.sample(64)
            # print(len(state), state[0].shape)  # (batch_size, action_dim)
            old_value = self.critic(old_state).squeeze()

            # 计算下一个价值
            next_value = self.critic(new_state).squeeze()
            last_value = (1 - done) * next_value

            # 计算优势和目标价值
            advantages = reward + self.gamma * last_value - old_value
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            advantages = advantages.detach()

            # 计算logprobs和熵
            # print(old_state.shape)  # (batch_size, action_dim)
            # print(self.critic(old_state).shape)  # (batch_size, 1)
            dist = Categorical(self.actor(old_state))
            # print(action.shape)  # (batch_size)
            logprob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            # 计算策略损失
            ratio = torch.exp(logprob - old_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * entropy

            # 计算价值损失
            critic_loss = nn.MSELoss()(old_value, advantages)

            # Backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
