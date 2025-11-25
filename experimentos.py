import time            
import csv            
import statistics     
import random         
import matplotlib.pyplot as plt  

from maze import generate_maze, SIZE


# ================================================================
# BFS — Implementación usada para los experimentos
# ================================================================

from collections import deque

def bfs(maze):
    

    n, m = len(maze), len(maze[0])   
    start = (0, 0)               
    goal = (n - 1, m - 1)           

   
    queue = deque([(start, [start])])

  
    visited = set([start])

    nodes_explored = 0 

 
    while queue:
        (x, y), path = queue.popleft()
        nodes_explored += 1

        if (x, y) == goal:
            return path, nodes_explored

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < n and 0 <= ny < m:
                if maze[nx][ny] != "1" and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

    return None, nodes_explored


# ================================================================
# ALGORITMO GENÉTICO — Versión para experimentos
# ================================================================

MOVES = [(1,0), (-1,0), (0,1), (0,-1)]

POP_SIZE = 50         
MOVE_LEN = 60       
GENERATIONS = 500   

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


# ================================================================
# EJECUCIÓN AUTOMÁTICA DE LOS 20 EXPERIMENTOS
# ================================================================

def correr_experimentos(num_experimentos=20):


    resultados = []

    for i in range(num_experimentos):
        print(f"\n=== Experimento #{i+1} ===")


        maze = generate_maze(SIZE, SIZE)

        t0 = time.perf_counter()          
        path_bfs, nodes_bfs = bfs(maze)   
        t_bfs = time.perf_counter() - t0  

        bfs_exito = path_bfs is not None
        bfs_longitud = len(path_bfs) if bfs_exito else None

        
        t0 = time.perf_counter()
        path_gen, gens = genetic_algorithm(maze)
        t_gen = time.perf_counter() - t0

        gen_exito = path_gen is not None
        gen_longitud = len(path_gen) if gen_exito else None

        print(f"BFS → éxito={bfs_exito}, tiempo={t_bfs:.4f}s, longitud={bfs_longitud}")
        print(f"Genético → éxito={gen_exito}, tiempo={t_gen:.4f}s, longitud={gen_longitud}, generaciones={gens}")

        resultados.append({
            "exp": i+1,
            "bfs_exito": bfs_exito,
            "bfs_tiempo": t_bfs,
            "bfs_longitud": bfs_longitud,
            "bfs_nodos": nodes_bfs,
            "gen_exito": gen_exito,
            "gen_tiempo": t_gen,
            "gen_longitud": gen_longitud,
            "gen_generaciones": gens
        })

    return resultados


# ================================================================
# GUARDAR LOS RESULTADOS EN CSV + GENERAR GRÁFICAS
# ================================================================

def guardar_csv(resultados, nombre_archivo="resultados_experimentos.csv"):
 
    with open(nombre_archivo, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Exp",
            "BFS_Exito",
            "BFS_Tiempo",
            "BFS_Longitud",
            "BFS_NodosExplorados",
            "Gen_Exito",
            "Gen_Tiempo",
            "Gen_Longitud",
            "Gen_Generaciones"
        ])

        for r in resultados:
            writer.writerow([
                r["exp"],
                r["bfs_exito"],
                r["bfs_tiempo"],
                r["bfs_longitud"],
                r["bfs_nodos"],
                r["gen_exito"],
                r["gen_tiempo"],
                r["gen_longitud"],
                r["gen_generaciones"]
            ])

    print(f"\nCSV guardado como: {nombre_archivo}")


def generar_graficas(resultados):

    # =======================
    # EXTRACCIÓN DE DATOS
    # =======================

    bfs_tiempos = [r["bfs_tiempo"] for r in resultados]
    gen_tiempos = [r["gen_tiempo"] for r in resultados]

    bfs_exitos = [r["bfs_exito"] for r in resultados]
    gen_exitos = [r["gen_exito"] for r in resultados]

    bfs_longitudes = [r["bfs_longitud"] for r in resultados if r["bfs_longitud"] is not None]
    gen_longitudes = [r["gen_longitud"] for r in resultados if r["gen_longitud"] is not None]

    bfs_nodos = [r["bfs_nodos"] for r in resultados]

    gen_generaciones = [r["gen_generaciones"] for r in resultados]


    # =======================
    # → FILTROS PARA LA SCATTER
    # =======================

    gen_longitudes_filtrado = []
    gen_tiempos_filtrado = []

    for r in resultados:
        if r["gen_longitud"] is not None:
            gen_longitudes_filtrado.append(r["gen_longitud"])
            gen_tiempos_filtrado.append(r["gen_tiempo"])


    # =======================
    # MÉTRICAS GLOBALES
    # =======================

    import statistics

    bfs_tiempo_prom = statistics.mean(bfs_tiempos)
    gen_tiempo_prom = statistics.mean(gen_tiempos)

    bfs_tasa_exito = sum(bfs_exitos) / len(bfs_exitos)
    gen_tasa_exito = sum(gen_exitos) / len(gen_exitos)

    print("\n===== RESUMEN GLOBAL =====")
    print(f"Tiempo promedio BFS: {bfs_tiempo_prom:.4f} s")
    print(f"Tiempo promedio Genético: {gen_tiempo_prom:.4f} s")
    print(f"Tasa éxito BFS: {bfs_tasa_exito*100:.1f}%")
    print(f"Tasa éxito Genético: {gen_tasa_exito*100:.1f}%")

    if bfs_longitudes:
        print(f"Longitud promedio BFS: {statistics.mean(bfs_longitudes):.2f}")
    if gen_longitudes:
        print(f"Longitud promedio Genético: {statistics.mean(gen_longitudes):.2f}")


    # ==========================================================
    # 1. BARRA — TIEMPO PROMEDIO
    # ==========================================================
    plt.figure()
    plt.title("Tiempo promedio de ejecución")
    plt.bar(["BFS", "Genético"], [bfs_tiempo_prom, gen_tiempo_prom])
    plt.ylabel("Tiempo en segundos")
    plt.grid(axis="y", linestyle="--", alpha=0.5)


    # ==========================================================
    # 2. BARRA — TASA DE ÉXITO
    # ==========================================================
    plt.figure()
    plt.title("Tasa de éxito (%)")
    plt.bar(["BFS", "Genético"], [bfs_tasa_exito*100, gen_tasa_exito*100])
    plt.ylabel("Porcentaje")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)


    # ==========================================================
    # 3. DISPERSIÓN — LONGITUD VS TIEMPO DEL GENÉTICO (CORREGIDO)
    # ==========================================================
    plt.figure()
    plt.title("Dispersión — Longitud de la ruta vs Tiempo (Genético)")
    plt.scatter(gen_longitudes_filtrado, gen_tiempos_filtrado)
    plt.xlabel("Longitud de ruta")
    plt.ylabel("Tiempo (s)")
    plt.grid(linestyle="--", alpha=0.4)


    # ==========================================================
    # 4. HISTOGRAMA — TIEMPOS
    # ==========================================================
    plt.figure()
    plt.title("Distribución de tiempos")
    plt.hist(bfs_tiempos, alpha=0.7, label="BFS")
    plt.hist(gen_tiempos, alpha=0.7, label="Genético")
    plt.legend()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia")
    plt.grid(axis="y", linestyle="--", alpha=0.5)


    # ==========================================================
    # 5. BOXPLOT — COMPARACIÓN DE TIEMPOS
    # ==========================================================
    plt.figure()
    plt.title("Boxplot de tiempos BFS vs Genético")
    plt.boxplot([bfs_tiempos, gen_tiempos], labels=["BFS", "Genético"])
    plt.ylabel("Tiempo (s)")


    # ==========================================================
    # 6. SCATTER — GENERACIONES vs ÉXITO
    # ==========================================================
    plt.figure()
    plt.title("Generaciones consumidas por el Algoritmo Genético")
    plt.scatter(range(len(gen_generaciones)), gen_generaciones)
    plt.xlabel("Experimento")
    plt.ylabel("Generaciones")
    plt.grid(linestyle="--", alpha=0.5)


    # ==========================================================
    # 7. NODOS EXPLORADOS BFS
    # ==========================================================
    plt.figure()
    plt.title("Nodos explorados por BFS en cada experimento")
    plt.plot(bfs_nodos, marker="o")
    plt.xlabel("Experimento")
    plt.ylabel("Nodos explorados")
    plt.grid(linestyle="--", alpha=0.5)


    # ==========================================================
    # 8. CORRELACIÓN GENÉTICO: LONGITUD VS GENERACIONES
    # ==========================================================
    gen_longitudes_cor = []
    gen_gens_cor = []

    for r in resultados:
        if r["gen_longitud"] is not None:
            gen_longitudes_cor.append(r["gen_longitud"])
            gen_gens_cor.append(r["gen_generaciones"])

    if gen_longitudes_cor:
        plt.figure()
        plt.title("Correlación — Longitud vs Generaciones (Genético)")
        plt.scatter(gen_longitudes_cor, gen_gens_cor)
        plt.xlabel("Longitud de ruta")
        plt.ylabel("Generaciones usadas")
        plt.grid(linestyle="--", alpha=0.5)

    # Mostrar todas las figuras
    plt.show()




# ================================================================
# MAIN — EJECUCIÓN DIRECTA DEL ARCHIVO
# ================================================================
if __name__ == "__main__":
    # 1. Ejecutar los 20 experimentos
    resultados = correr_experimentos(num_experimentos=20)

    # 2. Guardar resultados en CSV
    guardar_csv(resultados)

    # 3. Generar gráficas de comparación
    generar_graficas(resultados)
