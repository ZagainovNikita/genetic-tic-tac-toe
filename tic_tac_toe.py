import numpy as np
import time


class TicTacToe:
    def __init__(self, size: int = 3, goal: int = 3):
        self.size = size
        self.goal = goal
        self.grid = np.zeros(shape=(self.size, self.size), dtype=np.float32)

        self.tic = 1.0
        self.tac = -1.0
        self.empty = 0

    def make_move(self, flag, pos):
        if self.grid[*pos] != 0.0:
            result = dict(
                grid=self.grid,
                done=1.0,
                score=flag*(-1),
                next_move=flag*(-1)
            )
            return result

        self.grid[*pos] = flag
        result = dict(
            grid=self.grid,
            done=self.is_done(),
            score=self.get_score(),
            next_move=flag*(-1)
        )
        return result

    def is_done(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i + self.goal <= self.size) and (self.check_vertical(i, j) != 0.0):
                    return 1.0
                if (j + self.goal <= self.size) and (self.check_horizontal(i, j) != 0.0):
                    return 1.0
                if (i + self.goal <= self.size) and \
                    (j + self.goal <= self.size) and \
                        (self.check_diagonal_neg_slope(i, j) != 0.0):
                    return 1.0
                if (i >= self.goal - 1) and \
                    (j + self.goal <= self.size) and \
                        (self.check_diagonal_pos_slope(i, j) != 0.0):
                    return 1.0
        return 0

    def get_score(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i + self.goal <= self.size) and (self.check_vertical(i, j) != 0.0):
                    return self.check_vertical(i, j)
                if (j + self.goal <= self.size) and (self.check_horizontal(i, j) != 0.0):
                    return self.check_horizontal(i, j)
                if (i + self.goal <= self.size) and \
                    (j + self.goal <= self.size) and \
                        (self.check_diagonal_neg_slope(i, j) != 0.0):
                    return self.check_diagonal_neg_slope(i, j)
                if (i >= self.goal - 1) and \
                    (j + self.goal <= self.size) and \
                        (self.check_diagonal_pos_slope(i, j) != 0.0):
                    return self.check_diagonal_pos_slope(i, j)
        return 0

    def check_horizontal(self, row, col):
        line_sum = np.sum(self.grid[row, col:col+self.goal])
        if line_sum == self.goal * self.tic:
            return self.tic
        if line_sum == self.goal * self.tac:
            return self.tac
        return 0.0

    def check_vertical(self, row, col):
        line_sum = np.sum(self.grid[row:row+self.goal, col])
        if line_sum == self.goal * self.tic:
            return self.tic
        if line_sum == self.goal * self.tac:
            return self.tac
        return 0.0

    def check_diagonal_neg_slope(self, row, col):
        line_sum = 0.0
        for i in range(self.goal):
            line_sum += self.grid[row + i, col + i]

        if line_sum == self.tic * self.goal:
            return self.tic
        if line_sum == self.tac * self.goal:
            return self.tac
        return 0.0

    def check_diagonal_pos_slope(self, row, col):
        line_sum = 0.0
        for i in range(self.goal):
            line_sum += self.grid[row - i, col + i]

        if line_sum == self.tic * self.goal:
            return self.tic
        if line_sum == self.tac * self.goal:
            return self.tac
        return 0.0
