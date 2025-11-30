import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """

    # Best Q-value for next state s'
    best_next_q = np.max(Q[sprime])

    # TD target : r + gamma * max_a' Q(s', a')
    td_target = r + gamma * best_next_q

    # TD error
    td_error = td_target - Q[s, a]

    # Q-learning update
    Q[s, a] = Q[s, a] + alpha * td_error

    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as input the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """

    # Exploration
    if np.random.rand() < epsilone:
        n_actions = Q.shape[1]
        return np.random.randint(n_actions)

    # Exploitation
    return int(np.argmax(Q[s]))


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01  # learning rate
    gamma = 0.8   # discount factor
    epsilon = 0.2 # exploration rate

    n_epochs = 20
    max_itr_per_epoch = 100
    rewards = []

    # ---------------- TRAINING ----------------
    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            # Choose action with epsilon-greedy
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            # Step in the environment (Gymnasium API)
            Sprime, R, terminated, truncated, info = env.step(A)
            done = terminated or truncated

            r += R

            # Q-learning update
            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # move to next state
            S = Sprime

            # stopping criterion
            if done:
                break

        print("episode #", e, " : r = ", r)
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards as a function of epochs
    plt.plot(rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Total reward per episode")
    plt.title("Q-learning on Taxi-v3")
    plt.show()

    print("Training finished.\n")

    # ---------------- EVALUATION ----------------
    n_eval_episodes = 10
    eval_rewards = []

    for ep in range(n_eval_episodes):
        S, _ = env.reset()
        done = False
        total_r = 0

        # Greedy policy (no exploration)
        while not done:
            A = int(np.argmax(Q[S]))
            Sprime, R, terminated, truncated, info = env.step(A)
            done = terminated or truncated

            total_r += R
            S = Sprime

        eval_rewards.append(total_r)
        print(f"Eval episode {ep} : r = {total_r}")

    print("Mean evaluation reward =", np.mean(eval_rewards))

    env.close()
