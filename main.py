import vmas
from algorithms import *

from PPOAgent import *


NUM_EPISODES = 3000
NUM_STEPS = 500
NUM_ENVS = 32
UPDATE_INTERVAL = 250


def main():
    # 创建 VMAS transport 场景环境
    env = vmas.make_env(
        scenario="transport",
        num_envs=1,
        device=device,
        continuous_actions=False,
        seed=42
    )
    env.reset()

    ippo = IPPO(env)
    ippo.train()


if __name__ == '__main__':
    main()