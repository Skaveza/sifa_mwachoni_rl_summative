import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    def __init__(self, env, learning_rate=3e-4, gamma=0.99):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropies = []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        self.entropies.append(entropy.item())
        return action.item(), log_prob

    def learn(self, total_timesteps=300000):
        episode_rewards = []
        for _ in range(total_timesteps // self.env.max_steps):
            state, _ = self.env.reset()
            log_probs = []
            rewards = []
            done = False
            
            while not done:
                action, log_prob = self.select_action(state)
                state, reward, done, truncated, _ = self.env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                
                if done or truncated:
                    break
            
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for log_prob, G in zip(log_probs, returns):
                policy_loss.append(-log_prob * G)
            policy_loss = torch.stack(policy_loss).sum()
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            episode_rewards.append(sum(rewards))
        return episode_rewards, self.entropies

    def save(self, path):
        torch.save(self.policy.state_dict(), path + ".pth")

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + ".pth"))
        return self

    def predict(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state)
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action, None
