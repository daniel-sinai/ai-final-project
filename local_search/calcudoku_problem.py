import numpy as np
from game_board.game_board import Board, BoardManager


class CalcudokuState:
    def __init__(self, board, value):
        self.board = board
        self.value = value


class CalcudokuProblem:
    def __init__(self, size, cages, operations):
        self.size = size
        self.cages = cages
        self.operations = operations

    def _num_conflicts_of_cell(self, i, j, board):
        counter = 0

        for k in range(self.size):
            if k != j and board.grid_[i, k] == board.grid_[i, j]:
                counter += 1
            if k != i and board.grid_[k, j] == board.grid_[i, j]:
                counter += 1

        for cage, operation in zip(self.cages, self.operations):
            if (i, j) in cage:
                if not BoardManager.is_cage_valid(board, cage, operation):
                    counter += 1
                break

        return counter

    def _get_successors(self, state):
        successors = []
        for i in range(self.size):
            for j in range(self.size):
                for val in range(1, self.size + 1):
                    if state.board.grid_[i, j] != val:
                        new_grid = state.board.grid_.copy()
                        new_grid[i, j] = val
                        new_board = Board(self.size, new_grid)
                        successors.append(
                            CalcudokuState(
                                new_board, self.num_of_conflicts(new_board)))
        return successors

    def get_initial_state(self):
        board = Board(self.size, np.random.randint(1, self.size + 1,
                                                   (self.size, self.size),
                                                   dtype=int))
        return CalcudokuState(board, self.num_of_conflicts(board))

    def num_of_conflicts(self, board):
        return sum(self._num_conflicts_of_cell(i, j, board)
                   for i in range(self.size) for j in range(self.size))

    def get_lowest_valued_successor(self, state):
        return min(self._get_successors(state), key=lambda s: s.value)
