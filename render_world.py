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


def move_ball_center(arr_center, speed):
    speed_x = speed[0]
    speed_y = speed[1]
    arr_center[0] = arr_center[0] + speed_x
    arr_center[1] = arr_center[1] + speed_y


# Initiate game
pygame.init()
game_title = 'Bouncing ball'
pygame.display.set_caption(game_title)

WIDTH = 600
HEIGHT = 600
BLACK = (0, 0, 0)
speed = [2, 3]
tup_game_size = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(tup_game_size)

# Draw game boarder
draw_boarder(screen, tup_game_size)
color = (192,192,192)
ball_center = [40, 30]
radius = 10
pygame.draw.circle(screen, color, ball_center, radius)

pos = np.array((40.0, 30.0))
v = np.array((0.1, 0.1))

dt = 10
while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    if ball_center[0] < 0 or ball_center[0] > WIDTH:
        speed[0] = -speed[0]
    if ball_center[1] < 0 or ball_center[1] > HEIGHT:
        speed[1] = -speed[1]
        
    move_ball_center(ball_center, speed)
    screen.fill(BLACK)
    draw_boarder(screen, tup_game_size)
    pygame.draw.circle(screen, color, ball_center, radius)
    pygame.display.update()
    #pygame.display.flip()
