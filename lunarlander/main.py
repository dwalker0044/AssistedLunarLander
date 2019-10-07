import numpy as np
from plotmap import plot_map
import matplotlib.pyplot as plt
from typing import Callable

# EPSILON is the exploration rate, the higher this is the more the agent will explore.
EPSILON = 0.6
# ALPHA and GAMMA relate to the learning update function.
ALPHA = 0.1
GAMMA = 0.6

# Total number of episodes to train for.
NUM_EPISODES = 2000
# Displays animation once per one hundred episodes.
RENDER_ANIMATION = False

from lunarlandercustom import LunarLanderCustom
from qtablediscrete import QTable1dDiscrete


def run(env: LunarLanderCustom,
        q_table: QTable1dDiscrete,
        state: np.array,
        epsilon: float = EPSILON):
    if np.random.uniform(0, 1) < epsilon:
        # Explore action space
        action = env.action_space.sample()
    else:
        # Exploit learned values
        action = np.argmax(q_table.lookup_state(state))

    next_state, reward, done, _ = env.step(action)
    reward = reward[1]

    q_table.update(state, next_state, action, reward)
    return done, reward, next_state


def run_episodes(env: LunarLanderCustom,
                 q_table: QTable1dDiscrete,
                 state_0: np.array,
                 render_pred: Callable[[int], bool],
                 n_episodes: int = NUM_EPISODES,
                 epsilon: float = EPSILON, ):
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
    env = LunarLanderCustom()

    # Create a q-table that captures one dimension (angle) of the state space.
    # Since angle is continuous it needs to be discretize as we cannot store an infinite number of states.
    x_bins = np.arange(-2 * np.pi, 2 * np.pi, (4 * np.pi) / 720)
    q_table = QTable1dDiscrete(4, x_bins, env.action_space.n)

    state = env.reset()

    # Now run the training episodes.
    rewards = run_episodes(env,
                           q_table,
                           state,
                           render_pred=lambda x: RENDER_ANIMATION and x % 100 == 0)

    print("-" * 20)

    # Run 5 episodes using only the solution, i.e 0 learning rate, so we can see the performance of our agent.
    run_episodes(env,
                 q_table,
                 state,
                 render_pred=lambda x: True,
                 n_episodes=5,
                 epsilon=0.0)

    env.close()

    plot_map(q_table.q_table)

    plt.figure(2)
    plt.plot(moving_average(rewards))
    plt.title("Reward per episode")
    plt.draw()
    plt.show()


if __name__ == "__main__":
    # all_envs = envs.registry.all()
    # for env in all_envs:
    #     print(env)
    main()
