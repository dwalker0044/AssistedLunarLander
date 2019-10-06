import gym
import math
import numpy as np
from plotmap import plot_map
import matplotlib.pyplot as plt

EPSILON = 0.3
ALPHA = 0.1
GAMMA = 0.6
NUM_EPISODES = 20000

RENDER_ANIMATION = False
from lunarlandercustom import LunarLanderCustom
from qtablediscrete import QTable2dDiscrete


def run(env, q_table, state, epsilon=EPSILON):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table.lookup_state(state))  # Exploit learned values

    next_state, reward, done, _ = env.step(action)
    reward = reward[1]

    q_table.update(state, next_state, action, reward)
    return done, reward, next_state


def run_episodes(env, q_table, state_0, n_episodes=NUM_EPISODES, epsilon=EPSILON, render_pred=None):
    rewards = np.zeros(n_episodes)
    episode_reward = 0
    episode_count = 0

    state = state_0
    loop_count = 0

    while True:
        if episode_count >= n_episodes:
            break

        done, reward, next_state = run(env, q_table, state, epsilon)

        loop_count += 1
        if loop_count > 1000:
            print("Terminating loop")
            done = True

        if done:
            rewards[episode_count] = episode_reward
            episode_count += 1

            print(f"Episode: {episode_count}  Reward: {episode_reward}")

            loop_count = 0
            episode_reward = 0
            state = env.reset()
        else:
            episode_reward += reward

            state = next_state

            if render_pred and render_pred(episode_count):
                env.render()

    return rewards


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main():
    # env = gym.make("LunarLander-v2")
    env = LunarLanderCustom()

    # The visible area of the map has been normalize to be between +/-1. We go up to +/-2 as the
    # lander can go off the screen.
    y_bins = np.arange(-5, 20, 0.5)
    x_bins = np.arange(-10, 10, 0.5)
    q_table = QTable2dDiscrete(3, 1, x_bins, y_bins, env.action_space.n)

    state = env.reset()

    rewards = run_episodes(env,
                           q_table,
                           state,
                           render_pred=lambda x: RENDER_ANIMATION and x % 100 == 0)
    print("------------------------------------")
    run_episodes(env,
                 q_table,
                 state,
                 n_episodes=5,
                 epsilon=0.0,
                 render_pred=lambda x: True)

    env.close()

    shape = (y_bins.shape[0], x_bins.shape[0])

    plot_map(q_table.q_table, shape)

    plt.figure(2)
    plt.plot(moving_average(rewards))
    plt.draw()
    plt.show()


if __name__ == "__main__":
    # all_envs = envs.registry.all()
    # for env in all_envs:
    #     print(env)
    main()
