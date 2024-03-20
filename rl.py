import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import swanlab

swanlab.init(
    experiment_name="rl",   
    description="first rl demo for swanlab",
    logdir="./swanlog",  # 日志目录
)  # 初始化swanlab

# 超参数
learning_rate = 0.0002
gamma = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        # 定义神经网络结构
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.softmax(self.fc2(x), dim=0)  # 使用softmax函数处理输出，得到动作概率
        return x

    def put_data(self, item):
        self.data.append(item)  # 将(s, a, r)的元组放入数据列表中

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]: # 反向迭代数据列表，计算累计回报
            R = r + gamma * R  # 计算累计回报
            loss = -torch.log(prob) * R  # 计算策略梯度
            # swanlab.log({"loss": loss.item()})
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新网络参数
        self.data = []  # 清空数据列表


def main():
    env = gym.make('CartPole-v1')  # 创建CartPole环境
    pi = Policy()  # 创建策略网络
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False

        while not done:  # CartPole-v1在500步后强制终止
            prob = pi(torch.from_numpy(s).float())  # 通过策略网络获取动作概率
            m = Categorical(prob)  # 创建服从给定概率分布的Categorical分布
            a = m.sample()  # 从Categorical分布中采样动作
            s_prime, r, done, truncated, info = env.step(a.item())  # 执行动作，获取下一个状态和奖励
            pi.put_data((r, prob[a]))  # 将(s, a, r)元组放入数据列表中
            s = s_prime  # 更新状态
            score += r  # 更新累计奖励

        pi.train_net()  # 训练策略网络

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))  # 打印平均分数
            swanlab.log({
                "episode": n_epi,
                "avg_score": score / print_interval
            })
            score = 0.0  # 重置累计奖励
    env.close()


# 程序入口
if __name__ == '__main__':
    main()