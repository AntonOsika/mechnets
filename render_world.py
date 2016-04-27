import pygame
import sys
from pygame.locals import *


def draw_boarder(screen, tup_game_size):

    x_max = tup_game_size[0]
    y_max = tup_game_size[1]
    width = 10
    rgb_stainless = (224, 223, 219)

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
pos = (40, 30)
radius = 10
pygame.draw.circle(screen, color, pos, radius)

while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
