import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

ACTION_TABLE = {
    0: [0, 0, 0], # NOOP
    1: [1, 0, 0], # LEFT (forward)
    2: [1, 0, 1], # UPLEFT (forward jump)
    3: [0, 0, 1], # UP (jump)
    4: [0, 1, 1], # UPRIGHT (backward jump)
    5: [0, 1, 0], # RIGHT (backward)
}
input_dim = 12
hidden_dim = 128
output_dim = len(ACTION_TABLE)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
        

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        super().__init__()
        self.actor = MLP(input_dim, hidden_dim, output_dim, dropout)
        self.critic = MLP(input_dim, hidden_dim, 1, dropout)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = self(state)

        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        real_action = ACTION_TABLE[action.item()]
        return real_action
    





# agent_right = ActorCritic(input_dim, hidden_dim, output_dim)
# agent_right.load_state_dict(torch.load('agent_right.pth'))
# _ = agent_right.eval()
#
# agent_left = ActorCritic(input_dim, hidden_dim, output_dim)
# agent_left.load_state_dict(torch.load('agent_left.pth'))
# _ = agent_left.eval()
#
