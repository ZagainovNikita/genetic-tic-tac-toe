import argparse
import pickle
from tic_tac_toe import TicTacToe
from neural_net import GeneticNet
from genetic_algorithm import train


def main():
    parser = argparse.ArgumentParser(
        description="Train Tic Tac Toe genetic algorithm.")
    parser.add_argument("--generations", type=int, default=500,
                        help="Number of generations.")
    parser.add_argument("--population_size", type=int, default=200,
                        help="Size of the population.")
    parser.add_argument("--grid_size", type=int, default=3,
                        help="Size of the game grid.")
    parser.add_argument("--game_goal", type=int, default=3,
                        help="Goal to win the game.")
    parser.add_argument("--games_per_chromosome_as_tic", type=int, default=20,
                        help="Number of games played per chromosome as Tic.")
    parser.add_argument("--n_matings", type=int, default=75,
                        help="Number of matings per generation.")
    parser.add_argument("--mutation_prob", type=float, default=0.1,
                        help="Probability of mutation.")
    parser.add_argument("--mutation_range", type=float, default=0.5,
                        help="Range of mutation.")
    parser.add_argument("--save_file", type=str, default="best_chromosomes.pkl",
                        help="Save path for best chromosomes")
    parser.add_argument("--select_topk", type=str, default=10,
                        help="How many best chromosomes to save.")

    args = parser.parse_args()

    best_chromosomes = train(
        args.generations,
        args.population_size,
        args.grid_size,
        args.game_goal,
        args.games_per_chromosome_as_tic,
        args.n_matings,
        args.mutation_prob,
        args.mutation_range,
        args.select_topk
    )

    with open(args.save_file, "w") as f:
        pickle.dump(best_chromosomes, f)


if __name__ == "__main__":
    main()
