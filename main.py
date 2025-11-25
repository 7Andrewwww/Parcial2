import pygame
from maze import maze, TILE, SIZE
from collections import deque
import random

# ==========================
# COLORES
# ==========================
WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
BLUE  = (0,0,255)
RED   = (255,0,0)
CYAN  = (0,255,255)
YELLOW = (255,255,0)

pygame.init()
screen = pygame.display.set_mode((SIZE * TILE, SIZE * TILE))
pygame.display.set_caption("Vista General — BFS y Genético")


# ==========================
# FUNCIONES DE DIBUJO
# ==========================
def draw_maze(screen_to_draw):
    for i,row in enumerate(maze):
        for j,cell in enumerate(row):
            color = WHITE
            if cell == "1": color = BLACK
            elif cell == "S": color = GREEN
            elif cell == "E": color = BLUE
            pygame.draw.rect(screen_to_draw, color, (j*TILE, i*TILE, TILE, TILE))

def draw_path(path, color, screen_to_draw):
    if not path: return
    for (i,j) in path:
        pygame.draw.rect(screen_to_draw, color, (j*TILE, i*TILE, TILE, TILE))


# ==========================
# BFS
# ==========================
def bfs(maze):
    n,m = len(maze), len(maze[0])
    start = (0,0)
    goal  = (n-1, m-1)
    queue = deque([(start, [start])])
    visited = {start}
    nodes = 0

    while queue:
        (x,y), path = queue.popleft()
        nodes += 1

        if (x,y) == goal:
            return path, nodes, True   # <--- AGREGADO

        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx, y+dy
            if 0<=nx<n and 0<=ny<m and maze[nx][ny]!="1" and (nx,ny) not in visited:
                visited.add((nx,ny))
                queue.append(((nx,ny), path+[(nx,ny)]))

    return None, nodes, False  # BFS falló


def diagnostico_bfs(maze):
    print("\n=== Diagnóstico BFS ===")
    paredes = sum(row.count("1") for row in maze)
    libres = SIZE*SIZE - paredes

    print(f"   - Celdas libres: {libres}")
    print(f"   - Paredes: {paredes}")

    ex,ey = SIZE-1, SIZE-1
    vecinos = [(ex+1,ey),(ex-1,ey),(ex,ey+1),(ex,ey-1)]
    vecinos_validos = [
        (x,y) for x,y in vecinos
        if 0<=x<SIZE and 0<=y<SIZE and maze[x][y]!="1"
    ]

    if not vecinos_validos:
        print("   👉 Meta completamente rodeada por paredes.")
        return "Meta rodeada por paredes"
    else:
        print("   👉 No existe un camino conectado hasta la meta.")
        return "No existe camino hasta la meta"


def show_failure_window_bfs(reason):
    fail_screen = pygame.display.set_mode((SIZE*TILE, SIZE*TILE))
    pygame.display.set_caption("BFS — Fallo Detectado")

    running = True
    font = pygame.font.SysFont("Arial", 24)

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        draw_maze(fail_screen)
        text = font.render("FALLO BFS: " + reason, True, RED)
        fail_screen.blit(text, (10, 10))
        pygame.display.flip()


def show_bfs_window(path):
    bfs_screen = pygame.display.set_mode((SIZE*TILE, SIZE*TILE))
    pygame.display.set_caption("BFS — Camino Encontrado")

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        draw_maze(bfs_screen)
        draw_path(path, RED, bfs_screen)
        pygame.display.flip()



# ==========================
# GENETIC ALGORITHM
# ==========================
MOVES = [(1,0),(-1,0),(0,1),(0,-1)]
POP_SIZE = 50
MOVE_LEN = 60
GENERATIONS = 200


def random_individual():
    return [random.choice(MOVES) for _ in range(MOVE_LEN)]


def simulate(ind, maze):
    x,y = 0,0
    path = [(x,y)]
    choques = 0
    rep = 0
    n = len(maze)

    for dx,dy in ind:
        nx,ny = x+dx, y+dy

        if not (0<=nx<n and 0<=ny<n) or maze[nx][ny]=="1":
            choques += 1
            continue

        if (nx,ny) == (x,y):
            rep += 1

        x,y = nx,ny
        path.append((x,y))

    return (x,y), path, choques, rep


def fitness(ind, maze):
    (x,y),_,_,_ = simulate(ind, maze)
    n = len(maze)
    dist = abs((n-1)-x) + abs((n-1)-y)
    return 1/(dist+1)


def crossover(a,b):
    idx = random.randint(0, MOVE_LEN-1)
    return a[:idx] + b[idx:]


def mutate(ind):
    if random.random() < 0.1:
        ind[random.randint(0, MOVE_LEN-1)] = random.choice(MOVES)
    return ind



def genetic_algorithm(maze):
    population = [random_individual() for _ in range(POP_SIZE)]

    best_failed_path = None
    best_failed_dist = 9999
    best_fail_info = None

    for gen in range(GENERATIONS):
        scored = sorted([(fitness(ind,maze), ind) for ind in population], reverse=True)
        _, best_ind = scored[0]

        end_pos, path, choques, rep = simulate(best_ind, maze)
        dist = abs((SIZE-1)-end_pos[0]) + abs((SIZE-1)-end_pos[1])

        # Guardar mejor intento fallido
        if dist < best_failed_dist:
            best_failed_dist = dist
            best_failed_path = path
            best_fail_info = (end_pos, choques, rep)

        if end_pos == (SIZE-1, SIZE-1):
            return path, gen, True, best_failed_path, best_fail_info

        selected = [ind for _,ind in scored[:10]]
        new_population = selected[:]

        while len(new_population) < POP_SIZE:
            a,b = random.sample(selected,2)
            new_population.append(mutate(crossover(a,b)))

        population = new_population

    return None, GENERATIONS, False, best_failed_path, best_fail_info


def diagnostico_genetico(end_pos, choques, rep):
    print("\n=== Diagnóstico Genético ===")
    dist = abs((SIZE-1)-end_pos[0]) + abs((SIZE-1)-end_pos[1])

    print(f" - Última posición: {end_pos}")
    print(f" - Distancia a meta: {dist}")
    print(f" - Choques: {choques}")
    print(f" - Bucles: {rep}")

    if choques > MOVE_LEN * 0.3:
        print(" 👉 Falló por múltiples choques.")
        return "Exceso de choques contra paredes"
    elif rep > MOVE_LEN * 0.3:
        print(" 👉 Cayó en bucles.")
        return "Exceso de pasos repetidos (bucles)"
    elif dist > SIZE//2:
        print(" 👉 No se acercó a la meta.")
        return "No evolucionó individuos cercanos a la meta"
    else:
        print(" 👉 Evolucionó, pero no llegó.")
        return "Insuficiente calidad evolutiva"


def show_genetic_window(path):
    gscreen = pygame.display.set_mode((SIZE*TILE, SIZE*TILE))
    pygame.display.set_caption("Genético — Camino Encontrado")

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        draw_maze(gscreen)
        draw_path(path, CYAN, gscreen)
        pygame.display.flip()


def show_genetic_failure_window(path, reason):
    gscreen = pygame.display.set_mode((SIZE*TILE, SIZE*TILE))
    pygame.display.set_caption("Genético — Mejor Intento Fallido")

    font = pygame.font.SysFont("Arial", 24)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        draw_maze(gscreen)
        draw_path(path, YELLOW, gscreen)

        text = font.render("FALLO GENÉTICO: " + reason, True, RED)
        gscreen.blit(text, (10, 10))
        pygame.display.flip()



# ==========================
# EJECUCIÓN
# ==========================
print("\nEjecutando BFS...")
path_bfs, nodes, ok_bfs = bfs(maze)
print("→ BFS encontró ruta?", ok_bfs)

if ok_bfs:
    show_bfs_window(path_bfs)
else:
    reason = diagnostico_bfs(maze)
    show_failure_window_bfs(reason)


print("\nEjecutando Algoritmo Genético...")
path_gen, gens, ok_gen, best_fail_path, fail_info = genetic_algorithm(maze)
print("→ Genético encontró ruta?", ok_gen, "| Generaciones:", gens)

if ok_gen:
    show_genetic_window(path_gen)
else:
    end_pos, choques, rep = fail_info
    reason = diagnostico_genetico(end_pos, choques, rep)
    show_genetic_failure_window(best_fail_path, reason)



# ==========================
# VISTA GENERAL
# ==========================
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    draw_maze(screen)
    draw_path(path_bfs, RED, screen)
    draw_path(path_gen, CYAN, screen)
    pygame.display.flip()

pygame.quit()
