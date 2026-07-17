"""
Inspired from the video https://www.youtube.com/watch?v=hlv79rcHws0 :
Proximal Policy Optimization (PPO) is Easy With PyTorch | Full PPO Tutorial from Machine Learning with Phil
"""
import os
import pickle
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []  
        self.rewards = []
        # ISSUE 3 FIXED: Split old single 'dones' list into explicit termination and truncation buffers
        self.terminateds = []
        self.truncateds = []
        self.batch_size = batch_size

    def generate_batches(self, next_value, next_terminated, next_truncated, gamma, gae_lambda):
        state_arr = np.array(self.states)    
        action_arr = np.array(self.actions)  
        prob_arr = np.array(self.probs)      
        val_arr = np.array(self.vals)        
        reward_arr = np.array(self.rewards)  
        terminated_arr = np.array(self.terminateds)      
        truncated_arr = np.array(self.truncateds)
        
        # Combine flags to define the absolute end of a physical episode loop
        done_arr = np.logical_or(terminated_arr, truncated_arr)
        next_done = np.logical_or(next_terminated, next_truncated)

        T_steps, num_envs, obs_dim = state_arr.shape
        act_dim = action_arr.shape[-1]

        flat_states = state_arr.reshape(T_steps * num_envs, obs_dim)
        flat_actions = action_arr.reshape(T_steps * num_envs, act_dim)
        flat_probs = prob_arr.reshape(T_steps * num_envs)
        flat_vals = val_arr.reshape(T_steps * num_envs)

        advantages = np.zeros((T_steps, num_envs), dtype=np.float32)
        lastgaelam = np.zeros(num_envs, dtype=np.float32)

        for t in reversed(range(T_steps)):
            if t == T_steps - 1:
                next_val = next_value
                # ISSUE 3 FIXED: Bootstrap value if truncated/timeout; drop future value to 0 if terminated/crashed
                next_non_terminal = 1.0 - next_terminated
                next_non_terminal_gae = 1.0 - next_done
            else:
                next_val = val_arr[t+1]
                next_non_terminal = 1.0 - terminated_arr[t]
                next_non_terminal_gae = 1.0 - done_arr[t]
                
            # Delta calculation uses next_non_terminal (keeps next_val alive during time truncation timeouts)
            delta = reward_arr[t] + gamma * next_val * next_non_terminal - val_arr[t]
            # GAE tracking uses next_non_terminal_gae to cut off advantage propagation across independent episodes
            lastgaelam = delta + gamma * gae_lambda * next_non_terminal_gae * lastgaelam
            advantages[t] = lastgaelam

        flat_advantages = advantages.reshape(T_steps * num_envs)
        flat_returns = flat_advantages + flat_vals
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        total_samples = T_steps * num_envs
        indices = np.arange(total_samples, dtype=np.int64)
        np.random.shuffle(indices)
        
        batches = [indices[i:i+self.batch_size] for i in range(0, total_samples, self.batch_size)]

        return flat_states, flat_actions, flat_probs, flat_vals, flat_advantages, flat_returns, batches
    
    def store_memory(self, state, action, probs, vals, reward, terminated, truncated):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        # ISSUE 3 FIXED: Separately store the flags from the step function
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)

    def clear_memory(self):
        self.states, self.probs, self.vals = [], [], []
        self.actions, self.rewards = [], []
        self.terminateds, self.truncateds = [], []

class ActorNetwork(nn.Module): 
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=1024, fc2_dims=512, fc3_dims=256, fc4_dims=128, checkpoint_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_dims[0], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, n_actions),
        )
        self.init_weights()
        self.log_std = nn.Parameter(T.ones(n_actions) * -2.0)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)
    
    def init_weights(self):
        for m in self.actor:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, state):
        mean = self.actor(state)
        std = T.exp(self.log_std)
        return Normal(mean, std)
    
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class CriticNetwork(nn.Module): 
    def __init__(self, input_dims, alpha, fc1_dims=1024, fc2_dims=512, fc3_dims=256, fc4_dims=128, checkpoint_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims[0], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, 1)
        )
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def init_weights(self):
        for m in self.critic:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state): return self.critic(state)
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class Agent:
    def __init__(self, 
                 input_dims, n_actions, num_envs=1, 
                 gamma=0.99, gae_lambda=0.95,
                 alpha_actor=0.0003, alpha_critic=0.0001,  
                 value_loss_scale=1.0, entropy_loss_scale=0.001, 
                 policy_clip=0.1, gradient_norm_clip=0.5, value_loss_clip=0.1, 
                 actor_fc1_dims=1024, actor_fc2_dims=512, actor_fc3_dims=256, actor_fc4_dims=128,
                 critic_fc1_dims=1024, critic_fc2_dims=512, critic_fc3_dims=256, critic_fc4_dims=128,
                 batch_size=64, N=2048, n_epochs=10, 
                 checkpoint_dir='tmp/ppo'):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.N = N
        self.value_loss_scale = value_loss_scale
        self.entropy_loss_scale = entropy_loss_scale
        self.gradient_norm_clip = gradient_norm_clip
        self.value_loss_clip = value_loss_clip
        self.batch_size = batch_size

        self.actor = ActorNetwork(n_actions, input_dims, alpha_actor, checkpoint_dir=checkpoint_dir, fc1_dims=actor_fc1_dims, fc2_dims=actor_fc2_dims, fc3_dims=actor_fc3_dims, fc4_dims=actor_fc4_dims)
        self.critic = CriticNetwork(input_dims, alpha_critic, checkpoint_dir=checkpoint_dir, fc1_dims=critic_fc1_dims, fc2_dims=critic_fc2_dims, fc3_dims=critic_fc3_dims, fc4_dims=critic_fc4_dims)
        self.memory = PPOMemory(batch_size)
        self.scaler = RunningStandardScaler(shape=input_dims)
        self.scaler_checkpoint_file = os.path.join(checkpoint_dir, 'scaler_torch_ppo.pkl')

    def remember(self, state, action, probs, vals, reward, terminated, truncated):
        self.memory.store_memory(state, action, probs, vals, reward, terminated, truncated)

    def choose_action(self, raw_observation, low, high, update_scaler=True):
        if update_scaler:
            self.scaler.update(raw_observation)
        scaled_obs = self.scaler.transform(raw_observation)

        state = T.tensor(scaled_obs, dtype=T.float32).to(self.actor.device)
        low_t = T.tensor(low, dtype=T.float32).to(self.actor.device)
        high_t = T.tensor(high, dtype=T.float32).to(self.actor.device)
        
        with T.no_grad():
            dist = self.actor(state)
            u = dist.rsample()
            a = T.tanh(u)
            value = self.critic(state).squeeze(-1)
            action = low_t + (high_t - low_t) * (a + 1) / 2.0
            
            # Removed the redundant Tanh Jacobian correction terms (-T.log(1 - a^2) - T.log(scale)).
            # Since u is stored in memory and evaluated in log_diff, these constant scaling terms cancel out 
            # perfectly in the PPO ratio. Removing them eliminates extreme gradient spikes near the bounds (-1, 1).
            log_prob = dist.log_prob(u).sum(dim=-1)

        return action.cpu().numpy(), u.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy(), scaled_obs
    
    def learn(self, raw_next_obs, next_terminated, next_truncated, low, high):
        scaled_next_obs = self.scaler.transform(raw_next_obs)
        next_val_tensor = T.tensor(scaled_next_obs, dtype=T.float32).to(self.critic.device)
        with T.no_grad():
            next_value = self.critic(next_val_tensor).squeeze(-1).cpu().numpy()

        # Generate un-batched flat arrays from memory
        states, actions_u, old_probs, values, advantages, returns, _ = \
            self.memory.generate_batches(next_value, next_terminated, next_truncated, self.gamma, self.gae_lambda)

        states_t = T.tensor(states, dtype=T.float32).to(self.actor.device)
        actions_u_t = T.tensor(actions_u, dtype=T.float32).to(self.actor.device)
        old_probs_t = T.tensor(old_probs, dtype=T.float32).to(self.actor.device)
        advantages_t = T.tensor(advantages, dtype=T.float32).to(self.actor.device)
        returns_t = T.tensor(returns, dtype=T.float32).to(self.actor.device)
        old_values_t = T.tensor(values, dtype=T.float32).to(self.actor.device)

        value_losses = []
        entropy_losses = []
        
        total_samples = states.shape[0]
        indices = np.arange(total_samples, dtype=np.int64)

        for _ in range(self.n_epochs):
            # Reshuffle and regenerate unique mini-batches EVERY epoch
            np.random.shuffle(indices)
            
            for i in range(0, total_samples, self.batch_size):
                batch = indices[i:i+self.batch_size]
                
                dist = self.actor(states_t[batch])
                critic_value = self.critic(states_t[batch]).squeeze(-1)

                u_batch = actions_u_t[batch]
                new_probs = dist.log_prob(u_batch).sum(dim=-1)

                log_diff = new_probs - old_probs_t[batch]
                prob_ratio = T.exp(log_diff)
                
                b_advantages = advantages_t[batch]
                weighted_probs = b_advantages * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * b_advantages
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                v_unclipped = (returns_t[batch] - critic_value) ** 2
                v_clipped = old_values_t[batch] + \
                    T.clamp(critic_value - old_values_t[batch], -self.value_loss_clip, self.value_loss_clip)
 
                v_clipped_loss = (returns_t[batch] - v_clipped) ** 2
                critic_loss = 0.5 * T.max(v_unclipped, v_clipped_loss).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                total_loss = actor_loss + self.value_loss_scale * critic_loss - self.entropy_loss_scale * entropy
                
                value_losses.append(critic_loss.item())
                entropy_losses.append(entropy.item())

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.gradient_norm_clip)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.gradient_norm_clip)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        return np.mean(value_losses), np.mean(entropy_losses)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        with open(self.scaler_checkpoint_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print("Models and Scaler saved successfully.")

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        if os.path.exists(self.scaler_checkpoint_file):
            with open(self.scaler_checkpoint_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Models and Scaler loaded successfully.")
        else:
            print("Warning: Scaler checkpoint file not found! Using initialized values.")

class RunningStandardScaler:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def transform(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)