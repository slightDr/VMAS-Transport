import vmas
import numpy as np
from tqdm import tqdm

from PPOAgent import *


NUM_EPISODES = 3000
NUM_STEPS = 500
NUM_ENVS = 32
UPDATE_INTERVAL = 250

# # save path
# current_path = os.path.dirname(os.path.realpath(__file__))
# save_path = current_path + "/models/"
# timestamp = time.strftime("%Y%m%d-%H%M%S")


def ppo_train(
    num_envs=1, num_agents=4, num_episodes=400, num_steps=50000
):
    """
    在 vmas 的 transport 场景中训练 PPO。

    Args:
        num_envs (int): 并行环境数量。
        num_agents (int): 每个环境中的智能体数量。
        num_episodes (int): 训练总轮数。
        num_steps (int): 每轮中单个环境中的最大时间步数。
    """
    # 创建 VMAS transport 场景环境
    env = vmas.make_env(
        scenario="transport",
        num_envs=num_envs,
        device=device,
        continuous_actions=False,
        # seed=42
    )
    env.reset()

    # 获取状态和动作维度
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n

    # 创建智能体
    agents = [PPOAgent(state_dim, action_dim) for _ in range(num_agents)]
    memories = [Memory() for _ in range(num_agents)]
    total_reward = 0

    for episode in range(num_episodes):
        states = env.reset()  # shape: (num_envs, num_agents, state_dim)
        episode_rewards = np.zeros((num_envs, num_agents))

        for step in range(num_steps):
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
            next_states, rewards, dones, infos = env.step(actions)
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
            if (step + 1) % UPDATE_INTERVAL == 0:
                for agent_id, agent in enumerate(agents):
                    agent.update_new(memories[agent_id])
        # for agent_id, agent in enumerate(agents):
        #     agent.update_new_new(memories[agent_id])

        # 打印每个环境的总奖励
        print(f"Episode {episode + 1}, Rewards: {sum(episode_rewards.mean(axis=0)) / num_steps}")


def main():

    ppo_train()


if __name__ == '__main__':
    main()