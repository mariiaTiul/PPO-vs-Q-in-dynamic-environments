import argparse
import os
from distutils.util import strtobool
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import vizdoom as vzd

def setup_vizdoom():
    
    game = vzd.DoomGame()
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(True)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_mode(vzd.Mode.PLAYER)
    game.set_living_reward(-1)
    game.init()
    
    return game

class CNNActorCritic(nn.Module):
    
    def __init__(self, image_hight: int, image_width: int, num_actions: int):
        
        super(CNNActorCritic, self).__init__()

        h = image_hight 
        w = image_width
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        h //=4
        w //=4
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4)
        h //=4
        w //=4
        
        self.shared_fc = nn.Linear(h * w * 16, 128)
        self.actor_fc = nn.Linear(128, num_actions)
        self.critic_fc = nn.Linear(128, 1)

    
    def forward(self, x):
        
        batch_size = x.size(0)
        
        x = self.conv1 (x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = x.view(batch_size, -1)
        
        x = F.relu(self.shared_fc(x))
        
        actor_output = self.actor_fc(x)  
        critic_output = self.critic_fc(x)  

        return actor_output, critic_output

class PPO:
    
    def __init__(self, env):
        self._init_hyperparameters()
        self.env = env
        
        screen_shape = self.env.get_state().screen_buffer.shape
        
        self.obs_dim = np.prod(screen_shape)
        self.act_dim = len(env.get_available_buttons()) # 3
        
        self.actor = CNNActorCritic(120, 160, 3)
        self.critic = CNNActorCritic(120, 160, 1)

        #self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        #self.critic = FeedForwardNN(self.obs_dim, 1)

        #self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        #self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        self.actions = [
            [True, False, False],  # MOVE_LEFT
            [False, True, False],  # MOVE_RIGHT
            [False, False, True],  # ATTACK
        ]

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5 # number of epochs 
        
        self.gamma = 0.95
        self.clip = 0.2
        
        self.lr = 0.0003
        

    def learn(self, total_timesteps):
        
        t_so_far = 0  
        
        while t_so_far < total_timesteps: # 10000
            
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout() 

            t_so_far += np.sum(batch_lens)
            
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            for _ in range(self.n_updates_per_iteration):
                
                V, current_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(current_log_probs - batch_log_probs) 
                
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios,1-self.clip, 1+self.clip) * A_k
                
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                self.critic_optim.zero_grad() 
                critic_loss.backward()  
                self.critic_optim.step()
                
            
    def rollout(self):
        # Since this is an on-policy algorithm, we'll need to collect a fresh batch
        # of data each time we iterate the actor/critic networks.

        batch_obs = []  # Observations collected this batch
        batch_acts = []  # Actions collected this batch
        batch_log_probs = []  # Log probabilities of each action taken this batch
        batch_rews = []  # Rewards: (number of episodes, number of timesteps per episode)
        batch_rtgs = []  # Rewards-To-Go of each timestep in this batch
        batch_lens = []  # Lengths of each episode this batch
        
        t = 0
        while t < self.timesteps_per_batch: # 1000 
            episode_rewards = []
            
            self.env.new_episode() # equivalent for reset 

            obs = self.env.get_state().screen_buffer
            obs = obs.transpose(2, 0, 1)  
            obs = obs / 255.0 
            obs = torch.tensor(obs, dtype=torch.float)
            
            for ep_t in range(self.max_timesteps_per_episode):  # 500
                t += 1

                batch_obs.append(obs)
                
                action_idx, log_prob = self.get_action(obs.unsqueeze(0)) 
                batch_acts.append(action_idx)
                
                action = self.actions[action_idx]
                reward = self.env.make_action(action) # like a step, which retrieves a reward from an action 
                if len(episode_rewards) % 100 == 0:
                    print(f"Intermediate reward (last 100 timesteps): {sum(episode_rewards[-100:])}")

                
                if self.env.is_episode_finished():
                    break

                episode_rewards.append(reward)
                batch_log_probs.append(log_prob)

                
                obs = self.env.get_state().screen_buffer
                obs = obs.transpose(2, 0, 1)  
                obs = obs / 255.0
                obs = torch.tensor(obs, dtype=torch.float)
                
            print("Total reward:", env.get_total_reward())

                    
            batch_lens.append(ep_t + 1)
            batch_rews.append(episode_rewards)
            


        batch_obs = torch.stack(batch_obs)
        #batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        #batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        if isinstance(batch_acts, torch.Tensor):
            batch_acts = batch_acts.clone().detach().long()
        else:
            batch_acts = torch.as_tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float) 
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens 



    def compute_rtgs(self, batch_rews):
        
        # rewards-to-go per episode in the batch 
        batch_rtgs = []

        for episode_reward in reversed(batch_rews):

            reward_to_go = 0
            
            for reward in reversed(episode_reward):
                reward_to_go = reward + reward_to_go * self.gamma
                batch_rtgs.insert(0, reward_to_go)
                
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        
        return (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-10)


    def evaluate(self, batch_obs, batch_acts):
        
        _, V = self.critic(batch_obs)
        V = V.squeeze()  

        batch_acts = torch.tensor(batch_acts, dtype=torch.long)

        logits, _ = self.actor(batch_obs)
        action_probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(batch_acts)

        return V, log_prob

    def get_action(self, obs):
        
        logits, _ = self.actor(obs)
        action_probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().item(), log_prob.detach()



if __name__ == "__main__":
    # Set up ViZDoom
    env = setup_vizdoom()

    print(f"Available actions: {env.get_available_buttons()}")
    
    state = env.get_state()
    screen_buffer = state.screen_buffer  # Get the screen buffer (image)

    print(type(screen_buffer))

    print(f"Screen Buffer Shape: {screen_buffer.shape}")


    # Initialize PPO
    model = PPO(env)

    # Train PPO
    model.learn(10000)

    # Close environment
    env.close()
