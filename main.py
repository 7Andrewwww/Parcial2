import pygame
from maze import maze, TILE, SIZE   # Solo importamos la matriz y las constantes
from collections import deque
import random

# ==========================
# COLORES
# ==========================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
CYAN = (0, 255, 255)

pygame.init()
screen = pygame.display.set_mode((SIZE * TILE, SIZE * TILE))
pygame.display.set_caption("BFS y Algoritmo Genético")


# ==========================
# FUNCIÓN LOCAL PARA DIBUJAR EL LABERINTO
# ==========================
def draw_maze_local():
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            color = WHITE
            if cell == "1": color = BLACK
            elif cell == "S": color = GREEN
            elif cell == "E": color = BLUE

            pygame.draw.rect(screen, color, (j*TILE, i*TILE, TILE, TILE))


def draw_path(path, color):
    if not path:
        return
    for (i, j) in path:
        pygame.draw.rect(screen, color, (j*TILE, i*TILE, TILE, TILE))


# ==========================
# BFS
# ==========================
def bfs(maze):
    n, m = len(maze), len(maze[0])
    start = (0, 0)
    goal = (n - 1, m - 1)

    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:
            return path

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m:
                if maze[nx][ny] != "1" and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

    return None


# ==========================
# ALGORITMO GENÉTICO
# ==========================
MOVES = [(1,0), (-1,0), (0,1), (0,-1)]
POP_SIZE = 50
MOVE_LEN = 60
GENERATIONS = 200

def random_individual():
    return [random.choice(MOVES) for _ in range(MOVE_LEN)]

def simulate(ind, maze):
    x, y = 0, 0
    path = [(x, y)]
    n = len(maze)

    for dx, dy in ind:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and maze[nx][ny] != "1":
            x, y = nx, ny
            path.append((x, y))

    return (x, y), path

def fitness(ind, maze):
    (x, y), _ = simulate(ind, maze)
    n = len(maze)
    dist = abs((n - 1) - x) + abs((n - 1) - y)
    return 1 / (dist + 1)

def crossover(a, b):
    idx = random.randint(0, MOVE_LEN - 1)
    return a[:idx] + b[idx:]

def mutate(ind):
    if random.random() < 0.1:
        ind[random.randint(0, MOVE_LEN - 1)] = random.choice(MOVES)
    return ind

def genetic_algorithm(maze):
    population = [random_individual() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        scored = sorted([(fitness(ind, maze), ind) for ind in population], reverse=True)
        _, best_ind = scored[0]

        end_pos, path = simulate(best_ind, maze)
        n = len(maze)

        if end_pos == (n - 1, n - 1):
            return path, gen

        selected = [ind for _, ind in scored[:10]]

        new_population = selected.copy()
        while len(new_population) < POP_SIZE:
            a, b = random.sample(selected, 2)
            new_population.append(mutate(crossover(a, b)))

        population = new_population

    return None, GENERATIONS


# ==========================
# EJECUTAR ALGORITMOS
# ==========================
print("\nEjecutando BFS...")
path_bfs = bfs(maze)
print("→ BFS encontró ruta?", path_bfs is not None)

print("\nEjecutando Algoritmo Genético...")
path_gen, gens = genetic_algorithm(maze)
print("→ Genético encontró ruta?", path_gen is not None, "| Generaciones:", gens)


# ==========================
# LOOP PRINCIPAL (ESTABLE)
# ==========================
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
            break

    if not running:
        break

    draw_maze_local()
    draw_path(path_bfs, RED)
    draw_path(path_gen, CYAN)

    pygame.display.flip()

pygame.quit()
