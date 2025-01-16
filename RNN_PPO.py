import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.hidden_states_c = []
        self.is_terminals = []
        self.count = 0
    
    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.hidden_states_c = []
        self.is_terminals = []
        self.count = 0



class ActorCritic(nn.Module):
    def __init__(self, args, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.args = args
        self.device = torch.device(args.cuda)
        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_dim = 64
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor

        self.actor = PPOActor(args).to(self.device)
 
        # critic
        self.critic = PPOCritic(args).to(self.device)
    
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, rnn_state, hidden_state):

        if self.has_continuous_action_space:
            action_mean, action_var = self.actor(state)
            cov_mat = torch.diag(action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)


        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        action = torch.clamp(action, -1, 1)
        state_val = self.critic(state, rnn_state)
        rnn_input = torch.cat([state, action], dim=1)
        rnn_state, hidden_state = self.critic.rnn(rnn_input, hidden_state)

        return action.detach(), action_logprob.detach(), state_val.detach(), rnn_state, hidden_state
    
    def evaluate(self, state, action, rnn_state, hidden_state):

        if self.has_continuous_action_space:
            action_mean, action_var = self.actor(state)
            
            # action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = list()
        rnn_states = list()
        rnn_states.append(rnn_state)
        action = action.unsqueeze(1)
        for i in range(self.args.wait_batch_size):
            state_values.append(self.critic(state[i], rnn_state))
            rnn_input = torch.cat([state[i], action[i]], dim=1)
            rnn_state, hidden_state = self.critic.rnn(rnn_input, hidden_state)
            rnn_states.append(rnn_state)

        rnn_states = torch.cat(rnn_states[:-1]).unsqueeze(1)
        state_values = self.critic(state, rnn_states)
        
        return action_logprobs, state_values, dist_entropy

class PPOCritic(nn.Module):
    def __init__(self, args, hidden_dim=64, layer_norm=True):
        super(PPOCritic, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(args.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim+args.rnn_hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)
        self.rnn_net = nn.GRU(input_size=args.state_dim+1, hidden_size=args.rnn_hidden_dim, batch_first=True)


        if layer_norm:
            self.layer_norm(self.fc1, std=1.0)
            self.layer_norm(self.fc2, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs, rnn_state):
        x = F.tanh(self.fc1(inputs))  #Change to Tanh?
        x = torch.cat([x, rnn_state], dim=-1)
        x = F.tanh(self.fc2(x))
        v = self.fc3(x)
        return v
    
    def rnn(self, rnn_input, hidden_state):
        rnn_input, hidden_state = self.rnn_net(rnn_input, hidden_state)
        return rnn_input, hidden_state

class PPOActor(nn.Module):
    def __init__(self, args, n_actions=1):
        super(PPOActor, self).__init__()
        self.args = args
        self.hidden_dim = 16
        self.fc1 = nn.Linear(args.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean_head = nn.Linear(self.hidden_dim, n_actions)
        self.var_head = nn.Linear(self.hidden_dim, n_actions)

    def forward(self, obs):
        x = F.tanh(self.fc1(obs))
        x = F.tanh(self.fc2(x))
        q = F.tanh(self.mean_head(x))
        var = F.softplus(self.var_head(x))
        return q, var


class PPO:
    def __init__(self, args, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr_a = lr_actor
        self.lr_c = lr_critic
        self.device = torch.device(args.cuda)
        self.buffer = RolloutBuffer()
        self.buffer_new = RolloutBuffer()

        self.policy = ActorCritic(args, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(args, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.state_dim = state_dim
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                # print("setting actor output action_std to min_action_std : ", self.action_std)
            # else:
            #     print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        # print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, rnn_state, hidden_state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, action_logprob, state_val, rnn_state, hidden_state = self.policy_old.act(state, rnn_state, hidden_state)
            return action.detach().cpu().numpy().flatten(), action_logprob, state, state_val, action, rnn_state, hidden_state
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val, rnn_state, hidden_state = self.policy_old.act(state, rnn_state, hidden_state)

            return action.item(), action_logprob, state, state_val, action, rnn_state, hidden_state

    def update(self, ppo_state, ppo_hidden):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        # old_hidden_states_c = torch.squeeze(torch.stack(self.buffer.hidden_states_c, dim=0)).unsqueeze(0).detach().to(self.device)
        # old_hidden_states = torch.zeros([old_state_values.shape, 8, self.hidden_space])

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, ppo_state, ppo_hidden)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # hidden_states_c = torch.squeeze(hidden_states_c)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def lr_decay(self, decay):
        lr_a_now = self.lr_a * decay
        lr_c_now = self.lr_c * decay
        self.optimizer.param_groups[0]['lr'] = lr_a_now
        self.optimizer.param_groups[1]['lr'] = lr_c_now
   
       


