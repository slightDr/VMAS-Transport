# Episode 1, Rewards: 4.5501708984375e-07
# Episode 2, Rewards: -1.6864013671875e-05
# Episode 3, Rewards: 1.9385986328125e-05
# Episode 4, Rewards: 0.0
# Episode 5, Rewards: -1.07659912109375e-05
# Episode 6, Rewards: 3.720306396484375e-05
# Episode 7, Rewards: 1.20123291015625e-05
# Episode 8, Rewards: 4.696044921875e-06
# Episode 9, Rewards: 1.7606201171875e-05
# Episode 10, Rewards: -2.26116943359375e-05
# Episode 11, Rewards: 0.0
# Episode 12, Rewards: -2.5347900390625e-06
# Episode 13, Rewards: -8.978271484375e-06
# Episode 14, Rewards: 0.0
# Episode 15, Rewards: 0.0
# Episode 16, Rewards: 0.0
# Episode 17, Rewards: -1.1673583984375e-05
# Episode 18, Rewards: 0.0
# Episode 19, Rewards: 0.0


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# import numpy as np
# from vmas import make_env
#
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
#
# # 定义策略网络
# class PolicyNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, action_dim)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.softmax(self.fc2(x), dim=-1)
#         return x
#
# # 定义价值网络
# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # 定义CPPO算法
# class CPPO:
#     def __init__(self, policy_net, value_net, gamma=0.99, K=4, eps_clip=0.2, lr=0.01):
#         self.policy_net = policy_net
#         self.value_net = value_net
#         self.gamma = gamma
#         self.K = K
#         self.eps_clip = eps_clip
#         self.optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
#
#     def select_action(self, state):
#         state = torch.FloatTensor(state).to(device).unsqueeze(0)
#         action_probs = self.policy_net(state)
#         dist = Categorical(action_probs)
#         action = dist.sample()
#         return action, dist.log_prob(action)
#
#     def update(self, memory):
#         # 转换列表为张量
#         old_states = torch.stack(memory.states).detach()
#         actions = torch.stack(memory.actions).detach()
#         old_logprobs = torch.stack(memory.logprobs).detach()
#         old_values = self.value_net(old_states).squeeze()
#         rewards = torch.stack(memory.rewards).detach()
#         new_states = torch.stack(memory.next_states).detach()
#         dones = torch.stack(memory.dones).detach().float()
#
#         # 计算下一个价值
#         next_values = self.value_net(new_states).squeeze()
#         dones = dones
#         last_values = (1 - dones) * next_values
#
#         # 计算优势和目标价值
#         advantages = rewards + self.gamma * last_values - old_values
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
#
#         # 更新策略和价值网络
#         for _ in range(self.K):
#             indices = np.random.choice(len(old_states), size=64, replace=False)
#             batch_states = old_states[indices]
#             batch_actions = actions[indices]
#             batch_logprobs = old_logprobs[indices]
#             batch_advantages = advantages[indices]
#
#             # 计算logprobs和熵
#             dist = Categorical(self.policy_net(batch_states))
#             logprobs = dist.log_prob(batch_actions)
#             entropy = dist.entropy().mean()
#
#             # 计算策略损失
#             ratio = torch.exp(logprobs - batch_logprobs)
#             surr1 = ratio * batch_advantages
#             surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
#             policy_loss = -torch.min(surr1, surr2).mean() + 0.01 * entropy
#
#             # 计算价值损失
#             values = self.value_net(batch_states).squeeze()
#             value_loss = (values - batch_advantages).pow(2).mean()
#
#             # 反向传播
#             self.optimizer.zero_grad()
#             (policy_loss + 0.5 * value_loss).backward()
#             self.optimizer.step()
#
# # 定义Memory类
# class Memory:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.next_states = []
#         self.dones = []
#         self.logprobs = []
#
#     def clear(self):
#         del self.states[:]
#         del self.actions[:]
#         del self.rewards[:]
#         del self.next_states[:]
#         del self.dones[:]
#         del self.logprobs[:]
#
# # 创建VMAS环境
# env = make_env(
#     scenario="transport",
#     num_envs=1,
#     device=device,
#     continuous_actions=False,
#     wrapper=None,
#     max_steps=None,
#     seed=42,
#     dict_spaces=False,
#     grad_enabled=False,
#     terminated_truncated=False,
# )
#
# # 初始化CPPO代理
# state_dim = env.observation_space[0].shape[0]
# action_dim = env.action_space[0].n
# num_agents = env.n_agents
# policy_net = PolicyNetwork(state_dim, action_dim)
# value_net = ValueNetwork(state_dim)
# agent = CPPO(policy_net, value_net, gamma=0.99, K=4, eps_clip=0.2, lr=0.01)
#
# # 训练循环
# num_episodes = 1000
# for episode in range(num_episodes):
#     state = env.reset()
#     done = False
#     memory = Memory()
#     episode_reward = 0
#
#     t = 0
#     while not done and t < 1000:
#         t += 1
#         actions, logprobs = [], []
#         for i in range(num_agents):
#             action, logprob = agent.select_action(state[i])
#             actions.append(action)
#             logprobs.append(logprob)
#         next_state, reward, dones, _ = env.step(actions)
#         episode_reward += sum(reward)
#         if sum(reward) != 0:
#             print(episode_reward)
#
#         # 存储转换
#         memory.states.append(state)
#         memory.actions.append(actions)
#         memory.rewards.append(reward)
#         memory.next_states.append(next_state)
#         memory.dones.append(dones)
#         memory.logprobs.append(logprobs)
#
#         state = next_state
#         done = dones.item()
#         # 更新代理
#         if done or t == 1000:
#             done = True
#             agent.update(memory)
#             memory.clear()
#
#     print(f'Episode {episode+1}, Reward: {episode_reward}')