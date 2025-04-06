import time
import numpy as np
import imageio
import os
from collections import deque
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

TIMEOUT = 60 * 10 # 2 minutes per algorithm
FRAME_DIR = "frames"
GIF_DIR = "gifs"
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

def get_neighbors(env, state):
    neighbors = []
    for action in range(env.action_space.n):
        transitions = env.P[state][action]
        for prob, next_state, reward, done in transitions:
            neighbors.append((next_state, action, reward, done))
    return neighbors

def branch_and_bound(env):
    start_state = env.reset()[0]
    goal_state = env.nrow * env.ncol - 1
    visited = set()
    queue = deque([(start_state, [], 0)])
    start_time = time.time()

    while queue:
        current_state, path, depth = queue.popleft()
        if time.time() - start_time > TIMEOUT:
            break
        if current_state == goal_state:
            return path, time.time() - start_time
        if current_state in visited:
            continue
        visited.add(current_state)
        for next_state, action, reward, done in get_neighbors(env, current_state):
            if next_state not in visited:
                queue.append((next_state, path + [action], depth + 1))

    return None, TIMEOUT

def ida_star(env):
    goal_state = env.nrow * env.ncol - 1
    start_state = env.reset()[0]
    start_time = time.time()

    def dfs(state, path, depth, limit, visited):
        if time.time() - start_time > TIMEOUT:
            return None
        if depth > limit:
            return None
        if state == goal_state:
            return path
        visited.add(state)
        for next_state, action, _, _ in get_neighbors(env, state):
            if next_state not in visited:
                result = dfs(next_state, path + [action], depth + 1, limit, visited)
                if result:
                    return result
        visited.remove(state)
        return None

    limit = 0
    while True:
        visited = set()
        result = dfs(start_state, [], 0, limit, visited)
        if result:
            return result, time.time() - start_time
        if time.time() - start_time > TIMEOUT:
            return None, TIMEOUT
        limit += 1

def record_path(env, path, algo_name, run_num):
    state = env.reset()[0]
    frames = [env.render()]
    for step_idx, action in enumerate(path):
        state, _, _, _, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        imageio.imwrite(f"{FRAME_DIR}/{algo_name}_run{run_num}_step{step_idx+1}.png", frame)
    gif_path = f"{GIF_DIR}/{algo_name}_run{run_num}.gif"
    imageio.mimsave(gif_path, frames, duration=0.5)

def evaluate_solver(env_name, solver_fn, runs=5):
    times = []
    env = FrozenLakeEnv(is_slippery=False, render_mode="rgb_array")
    print(f"Evaluating {solver_fn.__name__} on {env_name}")

    for i in range(runs):
        path, elapsed = solver_fn(env)
        times.append(elapsed)
        if path:
            print(f"Run {i+1}: Goal reached in {len(path)} steps, Time: {elapsed:.2f}s")
            record_path(env, path, solver_fn.__name__, i + 1)
        else:
            print(f"Run {i+1}: Goal not reached, Time: {elapsed:.2f}s")

    print(f"Average Time: {np.mean(times):.2f}s\n")

if __name__ == "__main__":
    evaluate_solver("FrozenLake", branch_and_bound)
    evaluate_solver("FrozenLake", ida_star)
