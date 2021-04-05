#IMPORTS
import torch
import random
import numpy as np
from collections import deque
from game import GameAI,Point
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from model import Linear_QNet, QTrainer
from GraphPlot import plot

#VARIABLES
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
num_episode = 10000
SHOW_EVERY = 200
ep_reward = []
plot_reward =[]
AVG_reward =[]
START_EPSILON_DECAYING = 0.001
END_EPSILON_DECAYING = num_episode // 2


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.gamma = 0.45 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(4, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # self.epsilon_decay_value = (self.epsilon)/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)
        self.epsilon_decay_value = 0.998
    #TO DO
    def get_state(self, game):
        drone = game.drone
        state = [
            drone.x,drone.y,
            game.man.x,game.man.y# [1,0],[0,0],[0,1]

        ]
        return np.array(state, dtype=int)

    
    # Random Moves: tradeoff exploration / exploitation
    def get_action(self, state,episode):
        
        if END_EPSILON_DECAYING >= episode>= START_EPSILON_DECAYING:
            self.epsilon *= self.epsilon_decay_value
            self.epsilon = max(START_EPSILON_DECAYING,self.epsilon)        
        final_move = [0,0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
       
    #Storing Memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    #TO DO
    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Updating Q Values
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


# MAIN FUNCTION (RunfromHere)
def train():

    agent = Agent()
    game = GameAI()
    
    for episode in range(num_episode):
        
        episode_reward = 0
        game.reset()
        done = False
        
        while not done:
           
            state_old = agent.get_state(game) # get old state
            
            final_move = agent.get_action(state_old,episode) # get move
            
            reward, done = game.play_step(final_move) # perform move and get new state
            
            episode_reward += reward
            
            state_new = agent.get_state(game)
        
            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
        

         # train long memory, plot result
            
            agent.n_games += 1            
            agent.train_long_memory()            
        print('Episode:', episode,'Reward', reward,'Game', agent.n_games)

        plot_reward.append(episode_reward)
        ep_reward.append(episode)
        average_reward = sum(plot_reward[-SHOW_EVERY:])/len(plot_reward[-SHOW_EVERY:])
        print('Avg Reward:', average_reward)
        AVG_reward.append(average_reward)
        plt.plot(AVG_reward)
        plt.show()
        plt.savefig('Reward Graph.png')        
          
if __name__ == '__main__':
    train()
    