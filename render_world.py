import pygame
import sys
import numpy as np
from pygame.locals import *


rgb_stainless = (224, 223, 219)
def draw_boarder(screen, tup_game_size):

    x_max = tup_game_size[0]
    y_max = tup_game_size[1]
    width = 10

    pygame.draw.line(screen, rgb_stainless, [0, 0], [x_max, 0], width)
    pygame.draw.line(screen, rgb_stainless, [x_max, 0], [x_max, y_max], width)
    pygame.draw.line(screen, rgb_stainless, [x_max, y_max], [0, y_max], width)
    pygame.draw.line(screen, rgb_stainless, [0, y_max], [0, 0], width)


# Initiate game
pygame.init()
game_title = 'Bouncing ball'
pygame.display.set_caption(game_title)

WIDTH = 600
HEIGHT = 600
tup_game_size = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(tup_game_size)

# Draw game boarder
draw_boarder(screen, tup_game_size)
color = (192,192,192)
radius = 10

pos = np.array((40.0, 30.0))
v = np.array((0.1, 0.1))

dt = 10
while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    
    pygame.draw.circle(screen, (0,0,0), pos.astype('int'), radius)
    pos += dt*v
    pygame.draw.circle(screen, color, pos.astype('int'), radius)
    pygame.display.update()
    pygame.time.wait(dt)
