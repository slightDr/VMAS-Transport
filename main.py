import vmas
from algorithms import *

from PPOAgent import *


NUM_EPISODES = 200
NUM_STEPS = 60000
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
    ippo.train(num_episodes=NUM_EPISODES, num_steps=NUM_STEPS)

    # cppo = CPPO(env, NUM_EPISODES, NUM_STEPS, UPDATE_INTERVAL)
    # cppo.train(num_episodes=NUM_EPISODES, num_steps=NUM_STEPS)
    #
    # mappo = MAPPO(env, NUM_EPISODES, NUM_STEPS, UPDATE_INTERVAL)
    # mappo.train(num_episodes=NUM_EPISODES, num_steps=NUM_STEPS)


if __name__ == '__main__':
    main()