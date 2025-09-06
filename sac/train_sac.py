import gymnasium as gym
import numpy as np
from agent import Agent

if __name__ == '__main__':
    env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

    n_episodes = 250

    best_score = -1
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode = 'human')

    for i in range(n_episodes):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
