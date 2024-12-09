import torch
from tqdm import tqdm


def train(env, device, init, actor, evaluator, advantage_computer, trainer, num_epochs):
    num_agents = env.n_agents
    # print(env.observation_space)
    obs_dim = env.observation_space[0].shape[0]  # 假设所有智能体的观察空间相同
    # print(obs_dim)

    # print(env.action_space)
    action_dim = env.action_space[0].n  # 假设所有智能体的动作空间相同
    # print(action_dim)

    # 初始化训练器
    trainers = init(obs_dim, action_dim, num_agents, device)

    # 训练循环
    for epoch in tqdm(range(num_epochs)):
        obs = env.reset()
        # print(obs)  # obs[agent_idx][env_idx]
        trajectories = [[] for _ in range(num_agents)]
        done = False
        epoch_reward = 0

        # 采样轨迹
        t = 0
        while not done and t < 1000:
            t += 1
            actions, log_probs = [], []
            for i in range(num_agents):
                action, log_prob, _ = actor(trainers, i, obs, device)
                actions.append(action)
                log_probs.append(log_prob)

            next_obs, rewards, dones, infos = env.step(actions)
            if sum(rewards) != 0:
                epoch_reward += sum(rewards)
                print(epoch_reward)
            for i in range(num_agents):
                trajectories[i].append((obs[i], actions[i], rewards[i], log_probs[i], dones.item()))
            obs = next_obs

            done = dones.item()

        # 更新策略
        for i in range(num_agents):
            obs, actions, rewards, log_probs, dones = zip(*trajectories[i])
            values = evaluator(trainers, i, obs, device)
            advantages, returns = advantage_computer(trainers, i, rewards, values, dones)

            trajectories[i] = (torch.tensor(obs[i], dtype=torch.float32).to(device),
                               torch.tensor(actions, dtype=torch.int64).to(device),
                               torch.tensor(log_probs, dtype=torch.float32).to(device),
                               advantages.to(device),
                               returns.to(device))
            trainer(trainers, i, trajectories)

        # 打印结果
        print(f"Epoch {epoch + 1}/{num_epochs}: Total Reward = {sum(rewards)}")

    env.close()


# # 训练主循环
# def train(env_name="transport", num_agents=3, num_epochs=1000, algorithm="IPPO"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 初始化环境
#     env = vmas.make(env_name, num_agents=num_agents, device=device, continuous_actions=False)
#     obs_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#
#     # 初始化策略
#     if algorithm == "CPPO":
#         policy = PPOActorCritic(obs_dim, action_dim).to(device)
#         optimizer = optim.Adam(policy.parameters(), lr=3e-4)
#         trainers = [PPOTrainer(policy, optimizer)]
#     elif algorithm == "MAPPO":
#         policy = PPOActorCritic(obs_dim, action_dim).to(device)
#         optimizer = optim.Adam(policy.parameters(), lr=3e-4)
#         trainers = [PPOTrainer(policy, optimizer) for _ in range(num_agents)]
#     elif algorithm == "IPPO":
#         policies = [PPOActorCritic(obs_dim, action_dim).to(device) for _ in range(num_agents)]
#         optimizers = [optim.Adam(p.parameters(), lr=3e-4) for p in policies]
#         trainers = [PPOTrainer(p, o) for p, o in zip(policies, optimizers)]
#
#     # 训练循环
#     for epoch in range(num_epochs):
#         obs = env.reset()
#         trajectories = [[] for _ in range(num_agents)]
#         done = False
#
#         # 采样轨迹
#         while not done:
#             actions, log_probs, values = [], [], []
#             for i in range(num_agents):
#                 if algorithm == "CPPO" or algorithm == "MAPPO":
#                     action, log_prob, _ = trainers[i].policy.act(torch.tensor(obs[i], dtype=torch.float32).to(device))
#                     actions.append(action)
#                     log_probs.append(log_prob)
#                 elif algorithm == "IPPO":
#                     action, log_prob, _ = trainers[i].policy.act(torch.tensor(obs[i], dtype=torch.float32).to(device))
#                     actions.append(action)
#                     log_probs.append(log_prob)
#             next_obs, rewards, dones, infos = env.step(actions)
#             for i in range(num_agents):
#                 trajectories[i].append((obs[i], actions[i], rewards[i], log_probs[i], dones[i]))
#             obs = next_obs
#             done = all(dones)
#
#         # 更新策略
#         for i in range(num_agents):
#             obs, actions, rewards, log_probs, dones = zip(*trajectories[i])
#             values = [trainers[i].policy.critic(torch.tensor(o, dtype=torch.float32).to(device)).item() for o in obs]
#             advantages, returns = trainers[i].compute_advantages(rewards, values, dones)
#             trajectories[i] = (torch.tensor(obs, dtype=torch.float32).to(device),
#                                torch.tensor(actions, dtype=torch.int64).to(device),
#                                torch.tensor(log_probs, dtype=torch.float32).to(device),
#                                advantages.to(device),
#                                returns.to(device))
#             trainers[i].train(trajectories[i])
#
#         # 打印结果
#         print(f"Epoch {epoch + 1}/{num_epochs}: Total Reward = {sum(rewards)}")
#
#     env.close()