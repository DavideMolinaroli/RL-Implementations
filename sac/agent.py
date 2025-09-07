import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, env, n_actions, input_dims, tau = 0.005, alpha = 0.0003, beta = 0.0003, gamma = 0.99,max_size=1000000, layer1_size = 256, layer2_size = 256, batch_size = 256, reward_scale = 2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions = n_actions, name = 'actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims=input_dims, n_actions=n_actions, name='critic1')
        self.critic_2 = CriticNetwork(beta, input_dims=input_dims, n_actions=n_actions, name='critic2')
        self.value = ValueNetwork(beta, input_dims=input_dims, name='value')
        
        # target_value is just an exponential moving average of the value network, used just to stabilize training.
        # so the value function is updated using the self.value network. Other operations use this target_value network.
        # other operations are just the estimate Q_hat(st,at) for the critic networks
        self.target_value = ValueNetwork(beta, input_dims=input_dims, name='target_value')
        
        self.scale = reward_scale
        self.update_network_parameters(tau = 1)

    def choose_action(self, observation):
        # print(observation)
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparametrize=False)

        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transitions(state, action, reward, new_state, done)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        # batch of transitions of size batch_size
        s_t, a_t, r_t, s_tplus1, done = self.memory.sample_buffer(self.batch_size)

        s_t = T.tensor(s_t, dtype = T.float).to(self.actor.device)
        a_t = T.tensor(a_t, dtype=T.float).to(self.actor.device)
        r_t = T.tensor(r_t, dtype = T.float).to(self.actor.device)
        s_tplus1 = T.tensor(s_tplus1, dtype = T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        value = self.value(s_t).view(-1)
        value_s_tplus1 = self.target_value(s_tplus1).view(-1) # target_value is the value function used to estimate the value of s_{t+1}
        value_s_tplus1[done] = 0.0

        # q values of the current policy because the action is sampled from the actor and not from the replay buffer
        # value and actor networks need to use the current policy
        a_curr_policy, log_probs = self.actor.sample_normal(s_t, reparametrize=False)
        log_probs = log_probs.view(-1)
        q1_curr_policy = self.critic_1.forward(s_t, a_curr_policy)
        q2_curr_policy = self.critic_2.forward(s_t, a_curr_policy)
        critic_value = T.min(q1_curr_policy, q2_curr_policy) # for overestimation bias
        q_st_a_curr_policy = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        v_st_hat = (q_st_a_curr_policy - log_probs).detach() # soft policy iteration (equations 3 of the SAC paper)
        value_loss = 0.5*F.mse_loss(value, v_st_hat)
        # carefull: this loss would put gradients also into the actors and critic because of q_st_a_curr_policy and log_probs.
        # so, putting the detach() makes it so that no gradients are computed further those variables.
        value_loss.backward() 
        self.value.optimizer.step()

        # Reparametrize = True because this needs to be differentiated (see eq. 12 and 13 of the SAC paper)
        # So this is the same code used for the value loss, but with differentiability
        a_curr_policy, log_probs = self.actor.sample_normal(s_t, reparametrize=True)
        log_probs = log_probs.view(-1)
        q1_curr_policy = self.critic_1.forward(s_t, a_curr_policy)
        q2_curr_policy = self.critic_2.forward(s_t,a_curr_policy)
        critic_value = T.min(q1_curr_policy, q2_curr_policy)
        q_st_a_curr_policy = critic_value.view(-1)

        # equation 12 of the SAC paper
        actor_loss = T.mean(log_probs - q_st_a_curr_policy)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # detach to value_s_tplus1 to not put gradients into the target_value network
        q_hat = self.scale*r_t + self.gamma*value_s_tplus1.detach() # bootstrapping: Q_hat(s_t,a_t) = r + scale*V(s_{t+1})
        # use state,action pairs from the replay buffer to update the Q networks
        # because they need to learn the general environment dynamics
        q1_old_policy = self.critic_1.forward(s_t,a_t).view(-1)
        q2_old_policy = self.critic_2.forward(s_t,a_t).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()