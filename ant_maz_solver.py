import time
import heapq
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# Define AntMaze Environment
class AntMazeEnv:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (9, 9)
        self.obstacles = set(
            [
                (1, 2), (2, 2), (3, 2),
                (3, 3), (3, 4), (3, 5),
                (5, 5), (6, 5), (7, 5),
                (7, 6), (7, 7), (7, 8)
            ]
        )

    def reset(self):
        return self.start

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and pos not in self.obstacles

    def is_done(self, pos):
        return pos == self.goal

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = []
        for dx, dy in directions:
            next_pos = (x + dx, y + dy)
            if self.is_valid(next_pos):
                neighbors.append(next_pos)
        return neighbors

# Branch and Bound

def branch_and_bound(env):
    start_time = time.time()
    start = env.reset()
    goal = env.goal
    heap = [(0, start, [start])]
    visited = set()

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        if env.is_done(current):
            return path, time.time() - start_time

        for neighbor in env.get_neighbors(current):
            if neighbor not in visited:
                heapq.heappush(heap, (cost + 1, neighbor, path + [neighbor]))

    return None, time.time() - start_time

# IDA*

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def ida_star(env):
    start_time = time.time()
    start = env.reset()
    goal = env.goal

    def dfs(path, g, bound):
        node = path[-1]
        f = g + heuristic(node, goal)
        if f > bound:
            return f
        if env.is_done(node):
            return path

        min_bound = float('inf')
        for neighbor in env.get_neighbors(node):
            if neighbor not in path:
                t = dfs(path + [neighbor], g + 1, bound)
                if isinstance(t, list):
                    return t
                if t < min_bound:
                    min_bound = t
        return min_bound

    bound = heuristic(start, goal)
    path = [start]
    while True:
        t = dfs(path, 0, bound)
        if isinstance(t, list):
            return t, time.time() - start_time
        if t == float('inf'):
            return None, time.time() - start_time
        bound = t

# Draw Maze

def draw_maze(env, path=None, filename=None):
    grid = [['.' for _ in range(env.width)] for _ in range(env.height)]
    for ox, oy in env.obstacles:
        grid[oy][ox] = '#'

    gx, gy = env.goal
    grid[gy][gx] = 'G'

    if path:
        for (x, y) in path:
            if (x, y) != env.start and (x, y) != env.goal:
                grid[y][x] = '*'

    sx, sy = env.start
    grid[sy][sx] = 'S'

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])

    for y in range(env.height):
        for x in range(env.width):
            color = 'white'
            if grid[y][x] == '#':
                color = 'black'
            elif grid[y][x] == 'S':
                color = 'blue'
            elif grid[y][x] == 'G':
                color = 'green'
            elif grid[y][x] == '*':
                color = 'red'
            ax.add_patch(plt.Rectangle((x, env.height - 1 - y), 1, 1, color=color))

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    if filename:
        plt.savefig(filename)
    plt.close()

# Create GIF

def create_gif(env, path, gif_name="path.gif"):
    frames = []
    for i in range(1, len(path) + 1):
        filename = f"frame_{i:03d}.png"
        draw_maze(env, path[:i], filename)
        frames.append(imageio.imread(filename))
        os.remove(filename)

    imageio.mimsave(gif_name, frames, fps=2)
    print(f"GIF saved as {gif_name}")

# Evaluation Function

def evaluate_solver(env, solver_fn, name, iterations=5):
    print(f"\nEvaluating {name} on AntMaze")
    for i in range(iterations):
        path, elapsed = solver_fn(env)
        if path:
            print(f"Run {i+1}: Path length = {len(path)} | Time = {elapsed:.3f}s")
            draw_maze(env, path, filename=f"{name}_Run{i+1}.png")
            create_gif(env, path, gif_name=f"{name}_Run{i+1}.gif")
        else:
            print(f"Run {i+1}: No path found | Time = {elapsed:.3f}s")

# Main

if __name__ == '__main__':
    env = AntMazeEnv()
    evaluate_solver(env, branch_and_bound, "BranchAndBound", iterations=5)
    evaluate_solver(env, ida_star, "IDAStar", iterations=5)
