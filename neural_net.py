import numpy as np
from functools import reduce


def mul(x, y): return x * y


class GeneticNet:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.dim = grid_size ** 2
        self.shape_w1 = (self.dim, self.dim * 2)
        self.shape_b1 = (self.dim * 2, 1)
        self.shape_w2 = (self.dim * 2, self.dim * 2)
        self.shape_b2 = (self.dim * 2, 1)
        self.shape_w3 = (self.dim * 2, self.dim)
        self.shape_b3 = (self.dim, 1)

        self.chromosome_size = \
            reduce(mul, self.shape_w1) + \
            reduce(mul, self.shape_b1) + \
            reduce(mul, self.shape_w2) + \
            reduce(mul, self.shape_b2) + \
            reduce(mul, self.shape_w3) + \
            reduce(mul, self.shape_b3)

    def extract_from_chromosome(self, chromosome):
        start_idx = 0
        chromosome = np.reshape(chromosome, (-1,))

        self.w1 = chromosome[start_idx: start_idx+reduce(
            mul, self.shape_w1)].reshape(self.shape_w1)
        start_idx += reduce(mul, self.shape_w1)

        self.b1 = chromosome[start_idx: start_idx+reduce(
            mul, self.shape_b1)].reshape(self.shape_b1)
        start_idx += reduce(mul, self.shape_b1)

        self.w2 = chromosome[start_idx: start_idx+reduce(
            mul, self.shape_w2)].reshape(self.shape_w2)
        start_idx += reduce(mul, self.shape_w2)

        self.b2 = chromosome[start_idx: start_idx+reduce(
            mul, self.shape_b2)].reshape(self.shape_b2)
        start_idx += reduce(mul, self.shape_b2)

        self.w3 = chromosome[start_idx: start_idx+reduce(
            mul, self.shape_w3)].reshape(self.shape_w3)
        start_idx += reduce(mul, self.shape_w3)

        self.b3 = chromosome[start_idx: start_idx+reduce(
            mul, self.shape_b3)].reshape(self.shape_b3)
        start_idx += reduce(mul, self.shape_b3)

    def get_chromosome(self):
        return np.concatenate([
            self.w1.reshape(-1),
            self.b1.reshape(-1),
            self.w2.reshape(-1),
            self.b2.reshape(-1),
            self.w3.reshape(-1),
            self.b3.reshape(-1)
        ], axis=0)

    def forward(self, grid):
        x = grid.reshape(-1, 1)
        x = self.w1.T @ x + self.b1
        x = np.tanh(x)
        x = self.w2.T @ x + self.b2
        x = np.tanh(x)
        x = self.w3.T @ x + self.b3
        idx = np.argmax(x)
        return (idx // self.grid_size, idx % self.grid_size)
