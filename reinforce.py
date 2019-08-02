#Based on Official Pytorch docs (https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)
#and modified for this Pendulum env by Keyu
#usage: python3 reinforce.py




import argparse
import gym
import numpy as np
from itertools import count
from pendulum import PendulumEnv
from math import exp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = PendulumEnv() #Binary reward
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(3, 128)
        self.mu = nn.Linear(128, 1)
        self.sigma = nn.Linear(128,1)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    mu, sigma = policy(state)
    m = Normal(torch.Tensor(mu),torch.Tensor(sigma))
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    accu_reward_s = []
    accu_reward = 0
    for i_episode in range(5000):
        state = env.reset()
        for t in range(1000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step([action])
            if args.render:
                env.render()
            policy.rewards.append(reward)
            accu_reward += reward
            if done:
                break

        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tAccumulated Reward: {:f}\t'.format(
                i_episode, accu_reward))

        accu_reward_s.append(accu_reward)
        accu_reward = 0

    plt.figure()
    x = range(len(accu_reward_s))
    plt.scatter(x, accu_reward_s)
    plt.title('Maximum steps per episode:1000')
    plt.xlabel('i-th Episode')
    plt.ylabel('Accumulated reward per episode')
    plt.savefig('reinforce-1000*5000.png')

if __name__ == '__main__':
    main()
