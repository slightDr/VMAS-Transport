import vmas
from algorithms import *

from PPOAgent import *


NUM_EPISODES = 400
NUM_STEPS = 50000
UPDATE_INTERVAL = 250


def main():
    # 创建 VMAS transport 场景环境
    env = vmas.make_env(
        scenario="wheel",
        num_envs=1,
        device=device,
        continuous_actions=False,
        seed=42
    )
    env.reset()

    ippo = IPPO(env, NUM_EPISODES, NUM_STEPS, UPDATE_INTERVAL)
    ippo.train()

    # cppo = CPPO(env, NUM_EPISODES, NUM_STEPS, UPDATE_INTERVAL)
    # cppo.train()

    # mappo = MAPPO(env, NUM_EPISODES, NUM_STEPS, UPDATE_INTERVAL)
    # mappo.train()


if __name__ == '__main__':
    main()