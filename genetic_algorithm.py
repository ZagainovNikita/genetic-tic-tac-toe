import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from tic_tac_toe import TicTacToe
from neural_net import GeneticNet

matplotlib.use("TkAgg")


def init_population(population_size, grid_size):
    nets = [GeneticNet(grid_size) for _ in range(population_size)]
    chromosome_size = nets[0].chromosome_size
    population = np.random.normal(size=(population_size, chromosome_size))

    return population, nets


def play_game(player1, player2, size, goal):
    game = TicTacToe(size=size, goal=goal)
    done = 0.0
    grid = game.grid
    cur_player = 1.0

    while done == 0.0:
        if cur_player == 1.0:
            pos = player1.forward(grid)
        else:
            pos = player2.forward(grid * (-1))

        state = game.make_move(cur_player, pos)
        grid = state["grid"]
        done = state["done"]
        score = state["score"]
        cur_player = state["next_move"]

    if score == 1.0:
        return [1.75, 0.01]
    return [0.01, 2.25]


def fitness(nets, population, games_per_chromosome_as_tic, grid_size, game_goal):
    fitness_result = np.zeros(shape=(population.shape[0],), dtype=np.float32)
    for net, chromosome in zip(nets, population):
        net.extract_from_chromosome(chromosome)

    for p1_idx in range(population.shape[0]):
        for p2_idx in range(p1_idx + 1, p1_idx + (games_per_chromosome_as_tic // 2) + 1):
            player1 = nets[p1_idx]
            player2 = nets[p2_idx % population.shape[0]]

            results = play_game(player1, player2, grid_size, game_goal)
            fitness_result[p1_idx] += results[0]
            fitness_result[p2_idx % population.shape[0]] += results[1]

    return fitness_result


def crossover(population, fitness_result, n_matings):
    population_size, chromosome_length = population.shape
    weights = fitness_result / np.sum(fitness_result)
    offspring = np.zeros_like(population)

    for i in range(n_matings):
        parent1_idx = np.random.choice(population_size, p=weights)
        parent2_idx = np.random.choice(population_size, p=weights)

        crossover_point = np.random.randint(1, chromosome_length)

        offspring[i, :crossover_point] = population[parent1_idx, :crossover_point]
        offspring[i, crossover_point:] = population[parent2_idx, crossover_point:]

    for i in range(n_matings, population_size):
        best_idx = np.argmax(fitness_result)
        offspring[i] = population[best_idx]
        fitness_result[best_idx] = 0.0

    return offspring


def mutation(population, mutation_prob, mutation_range):
    number_of_mutations = int(population.size * mutation_prob)
    mutation_ids = np.random.randint(
        low=0,
        high=population.size-1,
        size=(number_of_mutations))

    for mutation_id in mutation_ids:
        mutation_value = np.random.uniform(
            low=-mutation_range, high=mutation_range, size=1)[0]

        population[mutation_id // population.shape[1],
                   mutation_id % population.shape[1]] += mutation_value

    return population


def train(
    generations,
    population_size,
    grid_size,
    game_goal,
    games_per_chromosome_as_tic,
    n_matings,
    mutation_prob,
    mutation_range
):
    plt.figure(figsize=(15, 9))

    population, nets = init_population(population_size, grid_size)

    max_score_history = np.zeros(generations)
    best_chromosome = np.zeros(nets[0].chromosome_size)
    best_score = 0

    loop = tqdm(range(generations))
    for generation in loop:
        loop.set_postfix_str("Mutating...")
        population = mutation(population, mutation_prob, mutation_range)

        loop.set_postfix_str("Evaluating...")
        fitness_score = fitness(
            nets, population, games_per_chromosome_as_tic, grid_size, game_goal)

        loop.set_postfix_str("Updating...")
        best_idx = np.argmax(fitness_score)
        max_score_history[generation] = fitness_score[best_idx]
        if fitness_score[best_idx] >= best_score:
            best_chromosome = np.copy(population[best_idx])
            best_score = fitness_score[best_idx]

        if generation < generations - 1:
            loop.set_postfix_str("Mating...")
            crossover(population, fitness_score, n_matings)

        loop.set_postfix_str("Plotting...")
        plt.clf()
        plt.plot(max_score_history[:generation+1], label="max score")
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(0.0000001)

    return best_chromosome


if __name__ == "__main__":
    train(
        generations=10000,
        population_size=70,
        grid_size=5,
        game_goal=4,
        games_per_chromosome_as_tic=30,
        n_matings=40,
        mutation_prob=0.05,
        mutation_range=1
    )
