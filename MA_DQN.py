import sys
from typing import Dict

# import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# from torch.utils.tensorboard import SummaryWriter

class DRQN():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.cuda)
        self.q = Q_net(args=args).to(self.device)   
        self.q_target = Q_net(args=args).to(self.device)  
        self.q_target.load_state_dict(self.q.state_dict())

        self.learn_step_counter = 0
        self.replace = 100
        self.optimizer = optim.Adam(self.q.parameters(), lr=args.lr)
        self.hidden_state = torch.zeros([1, args.rnn_input_size]).to(self.device)
        self.rnn_state = torch.zeros([1, args.rnn_input_size]).to(self.device)
        self.target_hidden_state = torch.zeros([1, args.rnn_input_size]).to(self.device)
        self.target_rnn_state = torch.zeros([1, args.rnn_input_size]).to(self.device)
        # self.target_hidden_state = torch.zeros([1, args.rnn_input_size]).to(self.device)

    def train(self, memory=None,
            device=None, 
            gamma=0.99):

        assert device is not None, "None Device input: device should be selected."

        # Get batch from replay buffer
        # samples, seq_len = episode_memory.sample()
        batch_size = self.args.batch_size

        observations = memory.obs
        actions = memory.action
        rewards = memory.reward
        next_observations = memory.next_obs
        dones = memory.done
        length = len(memory)

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        observations = torch.FloatTensor(observations.reshape(batch_size,-1)).to(self.device)
        actions = torch.LongTensor(actions.reshape(batch_size,-1)).to(self.device)
        rewards = torch.FloatTensor(rewards.reshape(batch_size,-1)).to(self.device)
        next_observations = torch.FloatTensor(next_observations.reshape(batch_size,-1)).to(self.device)
        dones = torch.FloatTensor(dones.reshape(batch_size,-1)).to(self.device)

        rnn_states = list()
        rnn_states.append(self.rnn_state.detach())
        target_rnn_states = list()
        for i in range(length):
            rnn_input = torch.cat([observations[i].detach(), actions[i].detach()]).unsqueeze(0)
            self.rnn_state, self.hidden_state = self.q.rnn(rnn_input, self.hidden_state)
            self.target_rnn_state, self.target_hidden_state = self.q_target.rnn(rnn_input.detach(), self.target_hidden_state)
            target_rnn_states.append(self.target_rnn_state.detach())
            if i != length-1:
                rnn_states.append(self.rnn_state.detach())
        

        # h_target = self.q_target.init_hidden_state(batch_size=batch_size, training=True)
        target_next_rnn_tensor = torch.cat(target_rnn_states)

        q_target = self.q_target(next_observations, target_next_rnn_tensor)

        q_target_max = q_target.max(1)[0].view(batch_size,-1).detach()
        targets = rewards + gamma*q_target_max*dones

    

        rnn_tensor = torch.cat(rnn_states)
        q_out = self.q(observations, rnn_tensor)
        q_a = q_out.gather(1, actions)

        # Multiply Importance Sampling weights to loss        
        loss = F.smooth_l1_loss(q_a, targets)
        
        # Update Network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.learn_step_counter % self.replace == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        self.learn_step_counter += 1

    def seed_torch(seed):
            torch.manual_seed(seed)
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    def save_model(model, path='default.pth'):
            torch.save(model.state_dict(), path)
 


# Q_network
class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()

        # space size check
        self.args = args
        self.hidden_space = 64
        self.device = torch.device(args.cuda)
        self.Linear1 = nn.Linear(args.state_dim, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.V = nn.Linear(self.hidden_space+args.rnn_hidden_dim, self.hidden_space)
        self.A = nn.Linear(self.hidden_space, args.action_dim)

        self.gru = nn.GRU(args.state_dim+1,args.rnn_hidden_dim, batch_first=True)
        self.Linear3 = nn.Linear(self.hidden_space, 1)
        

    def forward(self, state, rnn_state):
        x = F.relu(self.Linear1(state))
        x = F.relu(self.Linear2(x))
        x_v = torch.cat([x,rnn_state], dim=1)
        V = F.relu(self.V(x_v))
        V = self.Linear3(V)
        A = self.A(x)
        Q = V + (A - torch.mean(A, dim = 1, keepdim = True))
        return Q
    
    def advantage(self, state):
        x = F.relu(self.Linear1(state))
        x = F.relu(self.Linear2(x))
        return self.A(x)

    def rnn(self, x, hidden_state):
        rnn_state, hidden_state = self.gru(x, hidden_state)
        # rnn_state = F.relu(self.Linear3(rnn_state))
        return rnn_state, hidden_state

    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        with torch.no_grad():         
            advantage = self.advantage(obs)
            action = torch.argmax(advantage).item()

        if random.random() < epsilon:
            return random.randint(0,1)
        else:
            return action
        # return action

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, self.hidden_space])




class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)
    
    def clear(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []        

    def __len__(self) -> int:
        return len(self.obs)


