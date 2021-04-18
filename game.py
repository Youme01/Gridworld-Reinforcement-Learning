#IMPORTS
import pygame
import random
import math
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
Alpha = 0.9


class GameAI:

    def __init__(self, w=WIDTH, h=HEIGHT):

        self.w = w
        self.h = h
        self.distance = 0 
        self.display = pygame.display.set_mode((self.w, self.h)) # init display
        pygame.display.set_caption('SAR')
        #self.man_x,self.man_y = self._place_man()
        self.clock = pygame.time.Clock()
        self.reward_curr = 0
        self.reset()

    # Reset World
    def reset(self):
        self.drone = Point(self.w/2, self.h/2)

        self.man = Point(self.w-BLOCK_SIZE,self.h-BLOCK_SIZE)
        self._place_man()
        self.frame_iteration = 0

    # Man Position
    def _place_man(self):
        self.clock.tick(10)
        x = self.man.x
        y = self.man.y
        x -= 20  # man move constantly
        self.man = Point(x, y)
        return self.man
    
    # Updating reward after every step
    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move drone
        self._move(action)
        # 2. move drone
        self._move(action)
        if self.frame_iteration%3==0:
            self._place_man() 
        
        game_over = False

        distance = self.relative_distance()
        #print ("Distance",distance)
        
        # 4. reward for each step
        self.reward, game_over = self.get_reward()   
       
        # 5. update ui and clock
        #self._update_ui()

        self.clock.tick(80)
        return self.reward, game_over

    # Hits boundary
    def is_collision(self, pt=None):
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        return False
    
    def relative_distance(self):
        self.distance = math.floor(math.sqrt(((self.drone.x-self.man.x)**2) + ((self.drone.y - self.man.y)**2)))
        return self.distance
    
    def get_reward(self):
        """returns reward and done"""
        if self.is_collision(self.drone) or self.is_collision(self.man):
            return -300, True

        elif self.drone.x == self.man.x and self.drone.y == self.man.y:
            return 1000, True

        # elif (self.drone.y == self.man.y) and (self.drone.x != self.man.x):
        #     distance = self.relative_distance()
        #     r_curr =  math.log10(100 / (0.01*distance))
            
        #     return r_curr, False

        # elif (self.drone.y == self.man.y + 20) or (self.drone.y == self.man.y - 20):
        #     return 5, False

        else:
            return -1, False


    #TO DO
    # def get_reward(self, game_over):

    #     if (self.drone.x >= 0 and self.drone.x < 20) or (self.drone.x > self.w-BLOCK_SIZE and self.drone.x < self.w) :
    #         self.reward = -5
    #     else:
    #         self.reward = -1
    #     if (self.drone.y >= 0 and self.drone.y < 20) or (self.drone.y > self.h-BLOCK_SIZE and self.drone.y < self.h):
    #         self.reward = -5
    #     else:
    #         self.reward = -1
            
    #     if self.is_collision(self.drone) or self.frame_iteration > 100  : # colision & loop 
    #         game_over = True
    #         self.reward = -500 
    #         return self.reward, game_over
        
    #     if self.is_collision(self.man) : # colision & loop 
    #         game_over = True
    #         self.reward = -500
    #         return self.reward, game_over
        
    #     if self.drone == self.man: # place new man or just move
            
    #         self.reward = 3000
    #         self._place_man() 

        

    #     return self.reward, game_over

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