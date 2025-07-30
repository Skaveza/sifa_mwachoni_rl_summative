import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class REINFORCE:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        ).to(self.device)

    def predict(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state)
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action, None

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + ".pth"))
        self.policy.eval()
        return self