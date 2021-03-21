#IMPORTS
import pygame
import random

from collections import namedtuple
import numpy as np



#INITIALIZATION
pygame.init()
font = pygame.font.Font('arial.ttf', 25)
Point = namedtuple('Point', 'x, y')

# COLOR CODES
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

#VARIABLES
BLOCK_SIZE = 20
SPEED = 20
WIDTH = 200 
HEIGHT = 200


class GameAI:

    def __init__(self, w=WIDTH, h=HEIGHT):

        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h)) # init display
        pygame.display.set_caption('SAR')
        self.man_x,self.man_y = self._place_man()
        self.clock = pygame.time.Clock()
       
        self.reset()

    # Reset World
    def reset(self):
        self.drone = Point(self.w/2, self.h/2)
        self.score = 0
        self.man
        #self.man = Point(self.w-BLOCK_SIZE,self.h-BLOCK_SIZE)
        self._place_man()
        self.frame_iteration = 0

    # Man Position
    def _place_man(self):
        x =self.w -20
        y = self.h -20
        self.man = Point(x, y)
        return x ,y
    
    # Updating reward after every step
    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move drone
        self._move(action) 

        # 3. check if game over
        game_over = False

        # 4. reward for each step
        self.reward, game_over, self.score = self.get_reward(game_over)   
       
        # 5. update ui and clock
        self._update_ui()

        self.clock.tick(SPEED)

        return self.reward, game_over, self.score

    # Hits boundary
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.drone
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        return False

    #TO DO
    def get_reward(self, game_over):
        
        print(self.drone.x, self.drone.y)

        if (self.drone.x >= 0 and self.drone.x < 20) or (self.drone.x > self.w-BLOCK_SIZE and self.drone.x < self.w) :
            self.reward = -5
        else:
            self.reward = -1
        if (self.drone.y >= 0 and self.drone.y < 20) or (self.drone.y > self.h-BLOCK_SIZE and self.drone.y < self.h):
            self.reward = -5
        else:
            self.reward = -1
            
        if self.is_collision() or self.frame_iteration > 1000: # colision & loop 
            game_over = True
            self.reward = -500 
            return self.reward, game_over, self.score

        if self.drone == self.man: # place new man or just move
            self.score += 1
            self.reward = 1000
            self._place_man() 
        # elif pass:
        #     pass
        

        return self.reward, game_over, self.score

    #Update Display
    def _update_ui(self):
        self.display.fill(BLACK)

        pygame.draw.rect(self.display, RED, pygame.Rect(self.man.x, self.man.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.drone.x, self.drone.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()

    # Actions: based on direction updte the drone
    def _move(self, action):

        x = self.drone.x
        y = self.drone.y

        if np.array_equal(action, [1, 0, 0,0]):
            x += BLOCK_SIZE  # right
        elif np.array_equal(action, [0, 1, 0,0]):
            x -= BLOCK_SIZE # left
        elif np.array_equal(action, [0, 0, 1,0]): # [0, 0, 1]
            y += BLOCK_SIZE # down
        else:
            y -= BLOCK_SIZE # up
   
        self.drone = Point(x, y)