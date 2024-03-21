import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, trainsition):
        self.buffer.append(trainsition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        for trainsition in mini_batch:
            s, a, r, s_prime, done_mask = trainsition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

        return (
            torch.tensor(s_list, dtype=torch.float),
            torch.tensor(a_list),
            torch.tensor(r_list),
            torch.tensor(s_prime_list, dtype=torch.float),
            torch.tensor(done_mask_list),
        )

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 激活的目的是为了引入非线性
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.max(1)[1].item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    memory = ReplayBuffer()
    q = QNet()
    q_target = QNet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(
            0.01, 0.08 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if truncated else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0
        env.close()
        
if __name__ == "__main__":
    main()
