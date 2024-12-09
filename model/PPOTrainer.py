import torch


# PPO核心逻辑
class PPOTrainer:
    def __init__(self, policy, optimizer, clip_ratio=0.2, gamma=0.99, lam=0.95):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards) - 1)):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
        return advantages, returns

    def train(self, trajectories):
        obs, actions, log_probs_old, advantages, returns = trajectories
        for _ in range(10):  # 每个批次训练10次
            new_log_probs, entropy, values = self.policy.evaluate(obs, actions)
            ratio = (new_log_probs - log_probs_old).exp()

            # PPO损失
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # 值函数损失
            value_loss = ((returns - values) ** 2).mean()

            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()