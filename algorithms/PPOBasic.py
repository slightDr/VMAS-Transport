import torch
import numpy as np

from PPOAgent import Memory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOBasic:
    def __init__(self, env, num_episodes=400, num_steps=50000, update_interval=250):
        self.env = env
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.update_interval = update_interval

    def init_agents(self, n, state_dim, action_dim):
        pass

    def train(self, num_episodes=400, num_steps=5000):

        self.env.reset()

        # 获取状态和动作维度
        num_envs = self.env.num_envs
        num_agents = self.env.n_agents
        state_dim = self.env.observation_space[0].shape[0]
        action_dim = self.env.action_space[0].n

        # 创建智能体
        agents, memories = self.init_agents(num_agents, state_dim, action_dim)

        for episode in range(num_episodes):
            states = self.env.reset()  # shape: (num_envs, num_agents, state_dim)
            episode_rewards = np.zeros((num_envs, num_agents))

            for step in range(num_steps):
                # if step % 1000 == 0:
                #     print(f"step {step}")
                actions = []
                action_probs = []

                # 对每个智能体选择动作
                for agent_id, agent in enumerate(agents):
                    # 当前智能体的状态
                    states_array = np.array([s.cpu().numpy() for s in states])
                    # print(states_array.shape)  # (4, 32, 11)
                    agent_states = torch.tensor(states_array[agent_id, :, :], dtype=torch.float32).to(device)
                    # 选择动作
                    agent_actions = []
                    agent_probs = []
                    for env_idx in range(num_envs):
                        action, log_prob = agent.select_action(agent_states[env_idx])
                        agent_actions.append(action)
                        agent_probs.append(log_prob)

                    actions.append(agent_actions)
                    action_probs.append(agent_probs)

                # 与环境交互
                # print(len(actions), len(actions[0]))  # (4, num_envs)
                next_states, rewards, dones, infos = self.env.step(actions)
                dones = dones.to(dtype=torch.float32)  # 转换为浮点数以便后续计算

                # 存储每个智能体的交互数据
                # print(len(states), len(states[0]), len(states[0][0]))  # 4, num_envs, 11
                # print(len(rewards), len(rewards[0]))  # 4, num_envs
                for agent_id, memory in enumerate(memories):
                    for env_idx in range(num_envs):
                        memory.store(
                            states[agent_id][env_idx],
                            actions[agent_id][env_idx],
                            action_probs[agent_id][env_idx],
                            rewards[agent_id][env_idx],
                            dones[env_idx],
                            next_states[agent_id][env_idx],
                        )
                        episode_rewards[env_idx, agent_id] += rewards[agent_id][env_idx]

                states = next_states

                # 如果所有环境的所有智能体都完成了，提前结束
                if torch.all(dones):
                    break

            # 更新每个智能体的策略
            #     if (step + 1) % self.update_interval == 0:
            #         for agent_id, agent in enumerate(agents):
            #             agent.update_new(memories[agent_id])
            for agent_id, agent in enumerate(agents):
                agent.update(memories[agent_id])

            # 打印每个环境的总奖励
            print(f"Episode {episode + 1}, Rewards: {sum(episode_rewards.mean(axis=0)) / num_steps}")