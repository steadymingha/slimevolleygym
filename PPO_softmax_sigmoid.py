import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Bernoulli


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):#, has_continuous_action_space, action_std_init):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

        # self.actor = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     F.tanh(),
        #     nn.Linear(64, 64),
        #     F.tanh(),
        #     nn.Linear(64, action_dim),
        #     nn.Sigmoid()
        #     # nn.Softmax(dim=-1)
        # )
        # # critic
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     F.tanh(),
        #     nn.Linear(64, 64),
        #     F.tanh(),
        #     nn.Linear(64, 1)
        # )

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        action_ouput = self.fc3(x)

        forward_backward = F.softmax(action_ouput[..., :2], dim=-1)
        jump_flag = torch.sigmoid(action_ouput[..., 2:3])
        # raise NotImplementedError
        return torch.cat((forward_backward, jump_flag), dim=-1)

    def act(self, state):

        action_probs = self.forward(state)
        # dist = Categorical(action_probs)
        #
        # action = dist.sample()

        # dist = Bernoulli(action_probs)
        # action = dist.sample()

        # action_logprob = dist.log_prob(action)
        # action, action_logprob = self.process_softmax_output(action_probs)
        for_back_action, for_back_logprob = self.process_softmax_output(action_probs[:2])
        jump_action = (action_probs[-1] >= 0.5).float().unsqueeze(0)
        jump_logprob = torch.log(jump_action)
        jump_logprob = torch.where(jump_logprob==float('-inf'), torch.zeros_like(jump_logprob), jump_logprob)

        # state_val = self.critic(state)
        return torch.cat((for_back_action, jump_action)), \
                torch.cat((for_back_logprob, jump_logprob))
        # return action.detach(), action_logprob.detach()#, state_val.detach()

    def process_softmax_output(self, softmax_probs):
        # # 확률이 모두 0.5 미만인 경우 [0, 0] 반환
        # if torch.all(0.5 <= softmax_probs < 0.6):
        #     return torch.tensor([0, 0], dtype=torch.float32, device=softmax_probs.device), \
        #         torch.tensor([0, 0], dtype=torch.float32, device=softmax_probs.device)

        # 그렇지 않으면 확률이 더 높은 클래스를 1로 설정
        max_prob, max_index = torch.max(softmax_probs, dim=-1)
        output = torch.tensor([0, 0], dtype=torch.float32, device=softmax_probs.device)
        output[max_index] = 1

        # log_prob = torch.log(softmax_probs[max_index])
        log_prob = torch.log(output)
        log_prob = torch.where(log_prob==float('-inf'), torch.zeros_like(log_prob), log_prob)

        return output, log_prob

    # def evaluate(self, state, action):
    #
    #     # if self.has_continuous_action_space:
    #     #     action_mean = self.actor(state)
    #     #
    #     #     action_var = self.action_var.expand_as(action_mean)
    #     #     cov_mat = torch.diag_embed(action_var).to(device)
    #     #     dist = MultivariateNormal(action_mean, cov_mat)
    #     #
    #     #     # For Single Action Environments.
    #     #     if self.action_dim == 1:
    #     #         action = action.reshape(-1, self.action_dim)
    #     # else:
    #     action_probs = self.forward(state)
    #     # dist = Categorical(action_probs)
    #     dist = Bernoulli(action_probs)
    #
    #     action_logprobs = dist.log_prob(action)
    #     dist_entropy = dist.entropy()
    #     state_values = self.critic(state)
    #
    #     return action_logprobs, state_values, dist_entropy

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        # # critic
        # self.critic = nn.Sequential(
        #     F.linear(state_dim, 64),
        #     F.tanh(),
        #     F.linear(64, 64),
        #     F.tanh(),
        #     F.linear(64, 1)
        # )
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        critic_output = self.fc3(x)

        return critic_output



class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        # self.actor = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        # self.policy_old = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old = Actor(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.actor.state_dict())

        self.MseLoss = nn.MSELoss()

    def evaluate(self, state, action):
        action_probs = self.actor.forward(state)
        # dist = Categorical(action_probs)
        dist = Bernoulli(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic.forward(state)

        return action_logprobs, state_values, dist_entropy

    def select_action(self, state, action_idx):
        # if self.has_continuous_action_space:
        #     with torch.no_grad():
        #         state = torch.FloatTensor(state).to(device)
        #         action, action_logprob, state_val = self.policy_old.act(state)
        #
        #     self.buffer.states.append(state)
        #     self.buffer.actions.append(action)
        #     self.buffer.logprobs.append(action_logprob)
        #     self.buffer.state_values.append(state_val)
        #
        #     return action.detach().cpu().numpy().flatten()
        # else:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
            state_val = self.critic.forward(state).detach()

        if action_idx == 1:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action#.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            # logprobs, state_values, dist_entropy = self.actor.evaluate(old_states, old_actions)
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.unsqueeze(1)

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.actor.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))





