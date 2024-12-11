from PPOAgent import *
from .PPOBasic import PPOBasic


class MAPPO(PPOBasic):
    def __init__(self, env, num_episodes=400, num_steps=50000, update_interval=250):
        super().__init__(env, num_episodes, num_steps, update_interval)

    def init_agents(self, n, state_dim, action_dim):
        shared_actor = Actor(state_dim, action_dim).to(device)
        shared_critic = Critic(state_dim, action_dim).to(device)
        # 每个 agent 共享相同的 Actor 和 Critic
        agents = [PPOAgent(shared_actor, shared_critic) for _ in range(n)]
        # 每个 agent 拥有独立的 Memory
        memories = [Memory() for _ in range(n)]
        return agents, memories
