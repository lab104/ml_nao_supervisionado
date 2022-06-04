# Exercício 2 da Lista 3
#
# Tendo por base o exemplo DQN no
# CartPole Visual, gere uma versão 
# do código Actor-Critic no Cartpole 
# que receba como descrição do espaço
# de estados uma imagem, ao invés das 
# informações padrão da interface 
# do openai gym.

# Código Original: 
#   https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py#L39
#
# Renderizar Gym no Collab: 
#   https://stackoverflow.com/questions/50107530/how-to-render-openai-gym-in-google-colab


import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
import numpy as np 
import random
import math
            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1").unwrapped





# Este modelo vai aprender o que o ator
# deve fazer para maximizar o valor
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size        
        
        self.c1 = nn.Conv2d(3, 16, kernel_size=(6,6), stride=(1,1)) 
        self.c1d = nn.Dropout2d(0.25)
        self.c1p = nn.MaxPool2d(2)

        self.c2 = nn.Conv2d(16, 128, kernel_size=(6,6), stride=(1,1)) 
        self.c2d = nn.Dropout2d(0.25)
        self.c2p = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(128, 8, kernel_size=(6,6), stride=(1,1)) 
        self.c3d = nn.Dropout2d(0.25)

        self.f1 = nn.Flatten()                    

        self.d1 = nn.Linear(3384,2048)
        self.d2 = nn.Linear(2048,256)  
        self.o1 = nn.Linear(256, self.action_size)
    def forward(self, state):
        
        output = state
        output = F.relu(self.c1(output))        
        output = self.c1d(output)
        output = self.c1p(output)

        output = F.relu(self.c2(output))                
        output = self.c2d(output)
        output = self.c2p(output)

        output = F.relu(self.c3(output))                
        output = self.c3d(output)

        output = self.f1(output)

        output = F.relu(self.d1(output))
        output = F.relu(self.d2(output))

        output = self.o1(output)

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


# Função auxiliar que atenua a recompensa com a proximdade do fim do episódio
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


# Extrai imagem
image_width=600
image_height=50
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
    tela = tela / 255 - 0.5

    tela = torch.FloatTensor(tela)
    tela = tela.permute(0,3,1,2)    
    return tela

def captureScreen(env):
    tela = env.render(mode='rgb_array')
    tela = cv2.resize(tela,(60,40))
    return tela

def captureScreenRed(env):
    tela = np.zeros((40,60,3))
    tela[:,:,0] = 255
    return tela


import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('WebAgg')

def trainIters(actor, critic, n_iters):

    optimizerA = optim.Adam(actor.parameters(),lr=0.000001)
    optimizerC = optim.Adam(critic.parameters(),lr=0.0001)
    
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

        log_screens = []

        # executa o episodio
        for i in count():

            env.render()        
     
            state = torch.FloatTensor(state).to(device)            
            
            # consulta o ator para obter a informação da distribuição de probabilidades das ações
            dist = actor(tela)

            # consulta o crítico para obter a informação de valor do estado atual
            value = critic(state)
            
            # executa a próxima ação
            action = dist.sample()                        
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            if( iter % 10 == 0 ): log_screens.append( { 'i': i, 'action': str(action.cpu().numpy()[0]), 'tela': captureScreen(env), 'reward': reward } )

            next_tela = extractImage(env).to(device)
     
            # efetua os cálculos preliminares e armazena os resultados para o treinamento posterior
            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state
            tela = next_tela

            if done:
                if( iter % 10 == 0 ): log_screens.append( { 'i': i, 'action': 'Game Over', 'tela': captureScreenRed(env), 'reward': 0 } )
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

        # Terminou o episódio        
        # Treina o modelo    
         
        # Atualiza estado e valor            
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)

        # Aplica a lógica do gamma 
        returns = compute_returns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
                    
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean() 
                    # media quadráticas advantage
        critic_loss = advantage.pow(2).mean()    

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

        torch.cuda.empty_cache()

        if( iter % 10 == 0 ):            
            cols = 5
            rows = math.ceil(len(log_screens)/cols)+1
            if( rows != 0 ):                   
                
                fig, axs = plt.subplots(ncols=cols,nrows=rows,squeeze=True,figsize=(4*cols,3.5*rows))
                
                for row, axr in enumerate(axs):
                
                    for col, ax in enumerate(axr):

                        i = row*cols+col

                        if( i < len(log_screens)):
                            log_item = log_screens[i]
                            ax.imshow( log_item["tela"] )                    
                            ax.set_title( f'r={log_item["reward"]} a={log_item["action"]}')
                        
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        
                plt.tight_layout()
                plt.savefig('out/CartPole-4.png')
                plt.close()
                #plt.show()
      

    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':

    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')

    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')    

    actor = Actor(image_size,action_size).to(device)
    critic = Critic(state_size, action_size).to(device)        
    trainIters(actor, critic, n_iters=10000)