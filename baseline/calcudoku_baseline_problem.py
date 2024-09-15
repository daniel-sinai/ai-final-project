import numpy as np

from game_board.game_board import Board


class CalcudokuBaselineState:
    def __init__(self, grid):
        self.grid = grid

    def __eq__(self, other):
        if other is None or not isinstance(other, CalcudokuBaselineState):
            return False
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if self.grid[row, col] != other.grid[row, col]:
                    return False
        return True

    def __hash__(self):
        return hash(str(list(self.grid)))


class CalcudokuBaselineProblem:
    def __init__(self, board_manager):
        self.size = board_manager.size_
        self.board_manager = board_manager

    def get_initial_state(self):
        return CalcudokuBaselineState(np.zeros((self.size, self.size), dtype=int))

    def get_successors(self, state):
        successors = []
        for i in range(self.size):
            for j in range(self.size):
                if state.grid[i, j] == 0:
                    for val in range(1, self.size + 1):
                        new_grid = state.grid.copy()
                        new_grid[i, j] = val
                        successors.append(CalcudokuBaselineState(new_grid))
                    return successors
        return []

    def is_goal_state(self, state):
        return self.board_manager.is_goal_state(Board(self.size, state.grid))
