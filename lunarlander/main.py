import gym
import math
import numpy as np

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.6

bins = np.arange(-20, 20, 1)


def lookup_state(q_table, state):
    # Discretize the continuous position of the lander.
    x = np.digitize(state[0], bins)
    y = np.digitize(state[1], bins)

    state_index = y * bins.shape[0] + x
    return q_table[state_index]


def main():
    env = gym.make("LunarLander-v2")

    # The visible area of the map has been normalize to be between +/-1. We go up to +/-2 as the
    # lander can go off the screen.
    q_table = np.zeros([bins.shape[0] * bins.shape[0], env.action_space.n])

    state = env.reset()
    for _ in range(1000):
        if np.random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()  # Explore action space
        else:

            action = np.argmax(lookup_state(q_table, state))  # Exploit learned values

        next_state, reward, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()

        # old_value = lookup_action(q_table, state, action)
        old_state = lookup_state(q_table, state)
        old_value = old_state[action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        old_state[action] = new_value

        env.render()
        state = next_state
        # states from zero to seven.
        # pos x, pos y, vel x, vel y, lander angle, angular velocity, leg 0 ground contact, leg 1 ground contact.
        print(f"x: {state[0]:.4f}", end="")
        print(f"\t\ty: {state[1]:.4f}", end="")
        deg = state[4] * 180 / math.pi
        print(f"\t\tw: {deg:.4f}", end="")
        print(f"\t\tR: {reward:.4f}", end="\r")
        # print(f"\t\tw: {state[5]}", end="\r")

    env.close()


if __name__ == "__main__":
    # all_envs = envs.registry.all()
    # for env in all_envs:
    #     print(env)
    main()
