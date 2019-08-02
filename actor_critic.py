# Based on Official Pytorch docs (https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py)
# and modified for this Pendulum env by Keyu
# usage: python3 actor_critic.py


import argparse
import numpy as np
from collections import namedtuple
from pendulum import PendulumEnv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# some hyperparaters
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = PendulumEnv()
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(3, 128)
        self.l2 = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        mu = self.l2(x)
        sigma = self.l2(x)
        state_values = self.l2(x)
        return mu, sigma, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    mu, sigma, state_value = model(state)
    m = Normal(torch.Tensor(mu), torch.Tensor(sigma))
    action = m.sample()
    model.saved_actions.append(SavedAction(action, state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


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
            model.rewards.append(reward)
            accu_reward += reward
            if done:
                break
        accu_reward_s.append(accu_reward)
        finish_episode()

        # print accumulated reward every 10 episode
        if i_episode % args.log_interval == 0:
            print('Episode {} to {}:\tAccumulated Reward is {:5d}\t'.format(
                i_episode,i_episode+10,accu_reward))
        accu_reward = 0

    plt.figure()
    x = range(len(accu_reward_s))
    plt.scatter(x,accu_reward_s)
    plt.title('Maximum steps per episode:1000')
    plt.xlabel('i-th Episode')
    plt.ylabel('Accumulated reward per episode')
    plt.savefig('ac-1000*5000.png')

if __name__ == '__main__':
    main()
