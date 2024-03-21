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
    mutation_range,
    select_topk
):
    plt.figure(figsize=(15, 9))

    population, nets = init_population(population_size, grid_size)

    max_score_history = np.zeros(generations)
    running_record_history = np.zeros(generations)
    best_chromosomes = np.zeros(shape=(select_topk, nets[0].chromosome_size))
    best_scores = np.zeros(select_topk)

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

        top_ids_new = (-fitness_score).argsort()[:select_topk]
        best_scores = np.concatenate(
            [best_scores, fitness_score[top_ids_new]], axis=0)
        best_chromosomes = np.concatenate(
            [best_chromosomes, population[top_ids_new]], axis=0)

        top_ids_final = (-best_scores).argsort()[:select_topk]
        best_scores = best_scores[top_ids_final]
        best_chromosomes = best_chromosomes[top_ids_final]
        running_record_history[generation] = best_scores[0]

        if generation < generations - 1:
            loop.set_postfix_str("Mating...")
            crossover(population, fitness_score, n_matings)

        loop.set_postfix_str("Plotting...")
        plt.clf()
        plt.plot(max_score_history[:generation+1], label="max score")
        plt.plot(running_record_history[:generation+1], label="best score")
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(0.0000001)

    return best_chromosomes
