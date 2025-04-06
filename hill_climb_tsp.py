import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import random
import time

# Number of cities
NUM_CITIES = 20

def generate_cities(n):
    return np.random.rand(n, 2)

def total_distance(tour, cities):
    dist = 0
    for i in range(len(tour)):
        dist += np.linalg.norm(cities[tour[i]] - cities[tour[(i + 1) % len(tour)]])
    return dist

def plot_tour(cities, tour, filename):
    plt.figure(figsize=(8, 6))
    path = cities[tour + [tour[0]]]
    plt.plot(path[:, 0], path[:, 1], 'b-', marker='o')
    plt.title("Current Tour")
    plt.savefig(filename)
    plt.close()

def get_neighbor(tour):
    new_tour = tour.copy()
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def hill_climb(cities, max_iterations=500, save_gif=False, gif_name="hillclimb.gif"):
    current_tour = list(range(len(cities)))
    random.shuffle(current_tour)
    current_dist = total_distance(current_tour, cities)

    image_filenames = []

    for it in range(max_iterations):
        neighbor = get_neighbor(current_tour)
        neighbor_dist = total_distance(neighbor, cities)

        if neighbor_dist < current_dist:
            current_tour = neighbor
            current_dist = neighbor_dist

        if save_gif and it % 10 == 0:
            fname = f"frame_{it}.png"
            plot_tour(cities, current_tour, fname)
            image_filenames.append(fname)

    if save_gif:
        fname = f"frame_final.png"
        plot_tour(cities, current_tour, fname)
        image_filenames.append(fname)

        frames = []
        for fname in image_filenames:
            frames.append(imageio.imread(fname))
        imageio.mimsave(gif_name, frames, duration=0.3)

        for fname in image_filenames:
            os.remove(fname)

    return current_dist

# 🔥 MAIN FUNCTION with average stats
def run_multiple_trials(n_trials=5):
    total_time = 0
    total_dist = 0

    for i in range(n_trials):
        cities = generate_cities(NUM_CITIES)

        start_time = time.time()
        dist = hill_climb(cities, save_gif=True, gif_name=f"hillclimb_run_{i}.gif")
        end_time = time.time()

        elapsed = end_time - start_time
        total_time += elapsed
        total_dist += dist

        print(f"Run {i+1}: Final distance = {dist:.2f}, Time = {elapsed:.2f} seconds")

    avg_time = total_time / n_trials
    avg_dist = total_dist / n_trials
    print(f"\nAverage Time: {avg_time:.2f} seconds")
    print(f"Average Final Distance: {avg_dist:.2f}")

if __name__ == "__main__":
    run_multiple_trials()

