from PPOAgent import *
from .PPOBasic import PPOBasic


class CPPO(PPOBasic):
    def __init__(self, env, num_episodes=400, num_steps=50000, update_interval=250):
        super().__init__(env, num_episodes, num_steps, update_interval)

    def init_agents(self, n, state_dim, action_dim):
        actors = [Actor(state_dim, action_dim).to(device) for _ in range(n)]
        shared_critic = Critic(state_dim, action_dim).to(device)
        # 每个 agent 使用独立 Actor 和共享 Critic
        agents = [PPOAgent(actor, shared_critic) for actor in actors]
        memories = [Memory() for _ in range(n)]
        return agents, memories