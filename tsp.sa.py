import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# Number of places to visit
TOTAL_PLACES = 20

# Random (x, y) locations for each place
locations = np.random.rand(TOTAL_PLACES, 2) * 100

def get_path_length(path):
    """Calculate total path length for a list of places."""
    total = 0
    for i in range(len(path)):
        point1 = locations[path[i]]
        point2 = locations[(path[(i + 1) % len(path)])]
        total += np.linalg.norm(point1 - point2)
    return total

def make_new_path(path):
    """Create a slightly changed path by reversing a part of it."""
    a, b = sorted(random.sample(range(1, len(path)), 2))
    new_path = path[:a] + path[a:b+1][::-1] + path[b+1:]
    return new_path

def draw_path(path, heading="Travel Route", save_file=None):
    """Draw the travel route and save it as an image."""
    ordered_points = locations[path + [path[0]]]
    plt.figure(figsize=(8, 6))
    plt.plot(ordered_points[:, 0], ordered_points[:, 1], 'o-', color='green')
    for i, (x, y) in enumerate(locations):
        plt.text(x, y, str(i), fontsize=9, ha='right')
    plt.title(heading)
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_file:
        plt.savefig(save_file)
    plt.close()

def run_simulated_annealing(max_steps=1000, start_temp=100.0, cooling=0.995):
    """Main simulated annealing process."""
    start_time = time.time()

    current_path = list(range(TOTAL_PLACES))
    random.shuffle(current_path)
    current_length = get_path_length(current_path)

    best_path = current_path[:]
    best_length = current_length

    temperature = start_temp

    for _ in range(max_steps):
        new_path = make_new_path(current_path)
        new_length = get_path_length(new_path)

        change = new_length - current_length

        if change < 0 or random.random() < math.exp(-change / temperature):
            current_path = new_path
            current_length = new_length
            if current_length < best_length:
                best_path = current_path[:]
                best_length = current_length

        temperature *= cooling
        if temperature < 1e-5:
            break

    time_taken = time.time() - start_time
    return best_path, best_length, time_taken

if __name__ == "__main__":
    times_list = []
    lengths_list = []

    for run in range(5):
        print(f"\nTrial {run + 1}:")
        final_path, path_length, time_used = run_simulated_annealing()
        times_list.append(time_used)
        lengths_list.append(path_length)
        print(f"Time used: {time_used:.2f} seconds")
        print(f"Path length: {path_length:.2f}")
        draw_path(final_path, heading=f"Trial {run+1} - Distance: {path_length:.2f}", save_file=f"route_{run+1}.png")

    average_time = sum(times_list) / len(times_list)
    average_length = sum(lengths_list) / len(lengths_list)
    print(f"\nAverage Time: {average_time:.2f} seconds")
    print(f"Average Path Length: {average_length:.2f}")

