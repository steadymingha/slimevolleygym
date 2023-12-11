import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, dropout=0.1):
        super().__init__()
        self.actor = MLP(state_dim, hidden_dim, action_dim, dropout)
        self.critic = MLP(state_dim, hidden_dim, 1, dropout)

    def forward_rollout(self, state):
        output = self.actor(state)
        prob = F.softmax(output, dim=1)
        dist = Categorical(prob)

        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def forward_update(self, state, action):
        output = self.actor(state)
        prob = F.softmax(output, dim=1)
        dist = Categorical(prob)

        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy

    def forward_predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.forward_rollout(state)[0]

    def forward_value(self, state):
        value = self.critic(state)
        value = value.squeeze(dim=-1)
        return value

