import gym 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
import swanlab

run = swanlab.init(
    experiment_name="actor_critic",
    description="Actor Critic Algorithm with CartPole-v1 env",
    logdir="./swanlog",
)

# Hyperparameters
learning_rate = 0.002
# 折扣因子，用来平衡当前即时奖励和未来长期奖励的影响，当折扣因子接近于1时，算法倾向于长期奖励；当折扣因子接近于0时，算法倾向于即时奖励。
gamma = 0.98
n_rollout = 10
texts = []

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r/100.0])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
                                                               torch.tensor(r_list, dtype=torch.float), torch.tensor(s_prime_list, dtype=torch.float), \
                                                               torch.tensor(done_list, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    print_interval = 20
    score = 0.0
    
    for n_epi in range(10000):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))
                

                s = s_prime
                score += r
                
                if done:
                    break
                
            model.train_net()
            
        print(f"Episode: {n_epi}, Action: {a}, Reward: {r}, Done: {done}")
        text = swanlab.Text(f"Episode: {n_epi}, Action: {a}, Reward: {r}, Done: {done}", caption=f"{n_epi}")
        run.log({"examples": text})
        
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            swanlab.log({"epoch": n_epi, "avg_score": score/print_interval})
            score = 0.0
        env.close()
    
    print(texts)

    
if __name__ == '__main__':
    main()
            

            
            