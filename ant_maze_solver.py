import time
import heapq
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

TIMEOUT = 600  # seconds

def create_env():
    env = gym.make('AntMaze_UMaze-v5', render_mode='rgb_array')  # no 'human' mode unless GUI
    return env

def get_neighbors(state, env):
    neighbors = []
    for action in range(env.action_space.shape[0]):
        env.reset()
        env.set_state(state)
        new_state, _, terminated, truncated, _ = env.step(np.random.uniform(-1, 1, env.action_space.shape))
        done = terminated or truncated
        neighbors.append((new_state, action, done))
    return neighbors

def heuristic(state, goal):
    return np.linalg.norm(np.array(state) - np.array(goal))

def branch_and_bound(env):
    start_time = time.time()
    obs, _ = env.reset()
    start = env.get_state()
    goal = env.goal_pos

    visited = set()
    heap = [(0, start, [])]

    while heap:
        if time.time() - start_time > TIMEOUT:
            break

        cost, current_state, path = heapq.heappop(heap)
        env.set_state(current_state)
        obs, _, terminated, truncated, _ = env.step(np.zeros_like(env.action_space.sample()))
        if terminated or truncated:
            return path, time.time() - start_time

        visited.add(tuple(current_state))

        for next_state, action, done in get_neighbors(current_state, env):
            if tuple(next_state) not in visited:
                new_cost = cost + 1
                heapq.heappush(heap, (new_cost, next_state, path + [action]))

    return None, TIMEOUT

def ida_star(env):
    start_time = time.time()
    obs, _ = env.reset()
    start = env.get_state()
    goal = env.goal_pos

    def dfs(state, path, g, limit, visited):
        f = g + heuristic(state, goal)
        if f > limit:
            return None
        if time.time() - start_time > TIMEOUT:
            return None

        env.set_state(state)
        obs, _, terminated, truncated, _ = env.step(np.zeros_like(env.action_space.sample()))
        if terminated or truncated:
            return path

        visited.add(tuple(state))
        for next_state, action, done in get_neighbors(state, env):
            if tuple(next_state) not in visited:
                result = dfs(next_state, path + [action], g + 1, limit, visited.copy())
                if result is not None:
                    return result
        return None

    limit = heuristic(start, goal)
    while True:
        visited = set()
        result = dfs(start, [], 0, limit, visited)
        if result is not None:
            return result, time.time() - start_time
        if time.time() - start_time > TIMEOUT:
            break
        limit += 1
    return None, TIMEOUT

def evaluate_solver(env, solver_fn):
    times = []
    for i in range(1, 2):  # Reduce runs for now
        print(f"Run {i}: ", end="")
        path, elapsed = solver_fn(env)
        times.append(elapsed)
        if path:
            print(f"Goal reached in {len(path)} steps, Time: {elapsed:.2f}s")
        else:
            print(f"Goal not reached, Time: {elapsed:.2f}s")
    print(f"Average Time: {sum(times)/len(times):.2f}s")

if __name__ == "__main__":
    env = create_env()
    print("Evaluating branch_and_bound on AntMaze")
    evaluate_solver(env, branch_and_bound)
    print("\nEvaluating ida_star on AntMaze")
    evaluate_solver(env, ida_star)

