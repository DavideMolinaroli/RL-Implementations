import gymnasium as gym
import numpy as np
from ppo_discrete import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(
        n_actions=env.action_space.n,  # use env.action_space.n instead of hardcoding 1
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape
    )

    n_episodes = 300
    figure_file = 'plots/cartpole.png'

    best_score = -1
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_episodes):
        observation, info = env.reset()  # unpack obs and info
        done = False
        score = 0
        while not done:
            # I also need to store the value V of the observation to use it whenever this transition will be drawn from a sampled batch.
            # This will be needed to compute the reward-to-go (return) from that state after computing the advantages
            action, prob, val = agent.choose_action(observation)

            # Gymnasium step API
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            f'episode {i}, score {score:.1f}, avg score {avg_score:.1f}, '
            f'time_steps {n_steps}, learning_steps {learn_iters}'
        )
