import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOMemory:
    def __init__(self, batch_size):
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = [], [], [], [], [], []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = [], [], [], [], [], []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Parameter(T.zeros(n_actions))  # learnable log standard deviation

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.actor(state)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, 
                 batch_size=64, N=2048, n_epochs=10, action_bound=1.0):
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.action_bound = action_bound  # for tanh squashing

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... Saving Models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... Loading Models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actor.device)
        dist = self.actor(state)
        action = dist.sample()
        action_tanh = T.tanh(action) * self.action_bound
        value = self.critic(state)

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action_tanh.squeeze(0).cpu().detach().numpy(), log_prob.detach(), value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, old_log_probs, vals, rewards, dones, batches = self.memory.generate_batches()

            values = vals
            advantage = np.zeros(len(rewards), dtype=np.float32)

            # GAE computation (backward pass)
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * (1 - dones[t]) * (values[t+1] if t+1 < len(values) else 0) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantage[t] = gae

            advantage = T.tensor((advantage - advantage.mean()) / (advantage.std() + 1e-8)).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states_batch = T.tensor(states[batch], dtype=T.float).to(self.actor.device)
                actions_batch = T.tensor(actions[batch], dtype=T.float).to(self.actor.device)
                old_log_probs_batch = T.tensor(old_log_probs[batch], dtype=T.float).to(self.actor.device)

                dist = self.actor(states_batch)
                critic_value = self.critic(states_batch).squeeze()

                new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                prob_ratio = (new_log_probs - old_log_probs_batch).exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
