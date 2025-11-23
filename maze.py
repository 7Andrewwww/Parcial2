import pygame
import random

TILE = 40
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
SIZE = 16

pygame.init()
screen = pygame.display.set_mode((800, 800))


def generate_maze(n, m):
    maze = [["0" if random.random() > 0.25 else "1" for _ in range(m)] for _ in range(n)]
    maze[0][0] = "S"
    maze[n-1][m-1] = "E"
    return maze

maze = generate_maze(SIZE, SIZE)

def draw_maze():
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            color = WHITE
            if cell == "1": color = BLACK
            elif cell == "S": color = GREEN
            elif cell == "E": color = BLUE
            pygame.draw.rect(screen, color,
                             (j*TILE, i*TILE, TILE, TILE))

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    draw_maze()
    pygame.display.flip()

pygame.quit()