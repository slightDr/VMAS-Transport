from PPOAgent import *
from PPOBasic import PPOBasic


class IPPO(PPOBasic):
    def __init__(self, env):
        super().__init__(env)

    def init_agents(self, n, state_dim, action_dim):
        actors = [Actor(state_dim, action_dim) for _ in range(n)]
        critics = [Critic(state_dim, action_dim) for _ in range(n)]
        # 每个智能体单独拥有actor和critic
        agents = [PPOAgent(actor, critic) for actor, critic in zip(actors, critics)]
        memories = [Memory() for _ in range(n)]
        return agents, memories
