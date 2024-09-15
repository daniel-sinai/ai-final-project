import numpy as np


class Board:
    def __init__(self, size, grid=None):
        self.size_ = size
        if grid is None:
            self.grid_ = np.zeros((size, size), dtype=int)
        else:
            self.grid_ = grid


class BoardManager:
    def __init__(self, size, cages, operations):
        self.cages_ = cages
        self.size_ = size
        self.operations_ = operations

    @staticmethod
    def is_cage_valid(board, cage, operation):
        values = np.array(BoardManager._get_cage_values(board, cage))
        op = operation[0]
        if op == 'add':
            result = np.sum(values)
        elif op == 'subtract':
            result = np.abs(values[0] - values[1])
        elif op == 'multiply':
            result = np.prod(values)
        elif op == 'divide':
            result = np.max(values) // np.min(values)
        elif op == 'none':
            result = values[0]
        else:
            raise ValueError(f"Unknown operation: {operation[0]}")

        return result == operation[1]

    @staticmethod
    def _is_array_has_unique_values(arr):
        filtered_arr = arr[arr != 0]
        return len(np.unique(filtered_arr)) == len(filtered_arr)

    @staticmethod
    def _are_all_rows_unique(board):
        return np.all([BoardManager._is_array_has_unique_values(row)
                       for row in board.grid_])

    @staticmethod
    def _are_all_cols_unique(board):
        return np.all([BoardManager._is_array_has_unique_values(col)
                       for col in board.grid_.T])

    @staticmethod
    def _are_all_rows_and_cols_unique(board):
        return (BoardManager._are_all_rows_unique(board) and
                BoardManager._are_all_cols_unique(board))

    @staticmethod
    def _get_cage_values(board, cage):
        values = []
        for i, j in cage:
            values.append(board.grid_[i, j])
        return values

    def get_invalid_cages_num(self, board, check_zero_cages=True):
        invalid_cages_num = 0
        for cage, operation in zip(self.cages_, self.operations_):
            values = BoardManager._get_cage_values(board, cage)
            if 0 in values:
                if check_zero_cages:
                    invalid_cages_num += 1
                continue
            if not BoardManager.is_cage_valid(board, cage, operation):
                invalid_cages_num += 1
        return invalid_cages_num

    def _are_all_cages_valid(self, board):
        return self.get_invalid_cages_num(board) == 0

    def is_goal_state(self, board):
        return (
                not np.any(board.grid_ == 0) and
                self._are_all_cages_valid(board) and
                self._are_all_rows_and_cols_unique(board)
        )
