import torch
import random
from snek import Snake,  Direction, Point
from collections import deque
from model import Linear_QNET, QTrainer
from helper import plot
import numpy as np

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class SnakeAgent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount rate
        self.model = Linear_QNET(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=LR, gamma=self.gamma)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]


        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state,done):
        self.memory.append((state, action, reward, next_state, done)) #remove elemento à esquerda se atinge MAX_MEMORY
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #retorna uma lista de tuplas 
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
            
    def train_short_memory(self, state, action, reward, next_state,done):
        self.trainer.train_step(state, action, reward, next_state,done)
    
    def get_action(self, state):
        #random moves: tradeoff explorion / exploitation
        self.epsilon = 50 - self.n_games
        final_move = [0,0,0]

        if(random.randint(0, 200) < self.epsilon):

            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move



def train():
    scores = []
    mean_scores = []
    total_scores = 0
    record = 0
    agent = SnakeAgent()
    game = Snake()

    model = None

    while True:
        #get old state
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old)
        #perform mova and get new state
        reward, done, score, = game.play_step(final_move)
        state_new = agent.get_state(game)
   

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory
            #plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.n_games}, score: {score}, Record: {record}")           

            scores.append(score)
            total_scores+=score
            mean_score=total_scores/agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

                

if __name__ == '__main__':
    train()