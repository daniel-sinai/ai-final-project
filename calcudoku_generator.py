from game_generator.calcudoku_generator import Calcudoku


class CalcudokuBoard:
    def __init__(self, board, cages, operations):
        self.board_ = board
        self.cages_ = cages
        self.operations_ = operations

    @property
    def board(self):
        return self.board_

    @property
    def cages(self):
        return self.cages_

    @property
    def operations(self):
        return self.operations_

    def print(self):
        print(self.board_)
        print(self.cages_)
        print(self.operations_)


class CalcudokuGenerator:
    @staticmethod
    def generate(size):
        calc = Calcudoku()
        result = calc.generate(size)
        board = result.board.reshape(size, size)
        cages = []
        for cage in result.partitions:
            cages.append([(value//size, value % size) for value in cage])
        return CalcudokuBoard(board, cages, result.operations)
