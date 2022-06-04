# Código Original: https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py#L39
# Renderizar Gym no Collab: https://stackoverflow.com/questions/50107530/how-to-render-openai-gym-in-google-colab

import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
import numpy as np 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001



#class Actor(nn.Module):
#    def __init__(self, state_size, action_size):
#        super(Actor, self).__init__()
#        self.state_size = state_size
#        self.action_size = action_size
#        self.linear1 = nn.Linear(self.state_size, 128)
#        self.linear2 = nn.Linear(128, 256)
#       self.linear3 = nn.Linear(256, self.action_size)
#
#    def forward(self, state):
#        output = F.relu(self.linear1(state))
#        output = F.relu(self.linear2(output))
#        output = self.linear3(output)
#        distribution = Categorical(F.softmax(output, dim=-1))
#        return distribution


# Este modelo vai aprender o que o ator
# deve fazer para maximizar o valor
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size        
        self.l1 = nn.Conv2d(3, 64, kernel_size=2) # Saida = w/2       * h/2       * 64 
        self.l2 = nn.MaxPool2d(kernel_size=2)     # Saida = w/2/2     * h/2/2     * 64
        self.l3 = nn.Conv2d(64, 8, kernel_size=2) # Saida = w/2/2/2   * h/2/2/2   * 8
        self.l4 = nn.MaxPool2d(kernel_size=2)     # Saida = w/2/2/2/2 * h/2/2/2/2 
        self.l5 = nn.Flatten()                    
        self.l6 = nn.Linear(6912,256)
        self.l7 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.l1(state))
        output = self.l2(output)        
        output = F.relu(self.l3(output))
        output = self.l4(output)
        output = self.l5(output)
        output = F.relu(self.l6(output))
        output = self.l7(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

# O modelo vai aprender a pontuar a situação
# específica em que o modelo se encontra
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

image_width=150
image_height=100
image_channels=3
image_size=image_width*image_height*image_channels

def extractImage(env):
    tela = env.render(mode='rgb_array')
    tela = cv2.resize(tela,(image_width,image_height))
    #tela = np.dot(tela, [0.2989, 0.5870, 0.1140])

    #import matplotlib.pyplot as plt
    #plt.imshow(tela)
    #plt.savefig('teste.png')
    #plt.close()

    tela = tela.reshape(1,image_height,image_width,3)
    tela = tela / 255

    tela = torch.FloatTensor(tela)
    tela = tela.permute(0,3,1,2)    
    return tela

def trainIters(actor, critic, n_iters):

    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    
    for iter in range(n_iters):

        state = env.reset()
          
        tela = extractImage(env).to(device)
        #tela = torch.FloatTensor(tela).to(device)

        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        # executa uma interação, registra os dados para cada passo
        for i in count():
            env.render()          
     
            state = torch.FloatTensor(state).to(device)            
            #dist, value = actor(state), critic(state)
            dist, value = actor(tela), critic(state)
            #print(f'{dist} {dist.probs}')

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

            next_tela = extractImage(env).to(device)
            #next_tela = torch.FloatTensor(next_tela).to(device)
     
            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state
            tela = next_tela

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

        # calcula as losses e treina o modelo
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    #if os.path.exists('model/actor.pkl'):
    #    actor = torch.load('model/actor.pkl')
    #    print('Actor Model loaded')
    #if os.path.exists('model/critic.pkl'):
    #    critic = torch.load('model/critic.pkl')
    #    print('Critic Model loaded')    

    actor = Actor(image_size,action_size).to(device)
    critic = Critic(state_size, action_size).to(device)        
    trainIters(actor, critic, n_iters=10000)