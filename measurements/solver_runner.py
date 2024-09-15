import sys
import threading
from abc import ABC, abstractmethod
import statistics
import time
import numpy as np

from baseline.calcudoku_baseline_problem import CalcudokuBaselineProblem
from baseline.dfs_solver import DFSSolver
from csp.csp_solver import BacktrackingCSPSolver
from game_board.game_board import Board
from local_search.calcudoku_problem import CalcudokuProblem
from local_search.genetic_solver import GeneticSolver
from local_search.hill_climbing_solver import HillClimbingSolver

MEASUREMENTS = ["backtracks", "assignments", "board_validations"]
HILL_CLIMBING_MEASUREMENTS = ["conflicts"]
GENETIC_MEASUREMENTS = ["success", "generation"]


class SolverRunner(ABC):
    def __init__(self, measurements_names=MEASUREMENTS):
        self.__measurements = {m: [] for m in measurements_names}
        self.__lock = threading.Lock()

    @abstractmethod
    def get_name(self):
        pass

    def run(self, board_manager, timeout_seconds=None):
        self._solve_board(board_manager, timeout_seconds)

    def get_avg_for_measurements(self, name):
        if name not in self.__measurements.keys():
            raise ValueError(f"Measurement '{name}' is not valid. Must be one of {self.__measurements.keys()}.")

        measurements = self.__measurements[name]
        if not measurements:
            raise ValueError(f"Measurement {name} no found. {measurements}")

        return statistics.mean(measurements)

    def get_measurements(self, name):
        if name not in self.__measurements.keys():
            raise ValueError(f"Measurement '{name}' is not valid. Must be one of {self.__measurements.keys()}.")
        return self.__measurements[name]

    def reset_measurements(self):
        for name in self.__measurements.keys():
            self.__measurements[name].clear()

    @abstractmethod
    def _solve_board(self, board_manager, timeout_seconds=None):
        pass

    def _add_measurements(self, name, value):
        if name not in self.__measurements.keys():
            raise ValueError(f"Measurement '{name}' is not valid. Must be one of {self.__measurements.keys()}.")

        with self.__lock:
            self.__measurements[name].append(value)

    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time)
        return execution_time, result


class CSPSolverRunner(SolverRunner):
    def __init__(self, select_unassigned_var_type="none",
                 order_domain_values_type="random", use_fc=False, use_ac=False):
        super().__init__()
        self.use_fc = use_fc
        self.use_ac = use_ac
        self.select_unassigned_var_type = select_unassigned_var_type
        self.order_domain_values_type = order_domain_values_type

    def _solve_board(self, board_manager, timeout_seconds=None):
        solver = BacktrackingCSPSolver(board_manager, self.select_unassigned_var_type,
                                       self.order_domain_values_type,
                                       self.use_fc, self.use_ac)
        result, backtracks, assignments, board_validations = solver.solve(timeout_seconds)
        self._add_measurements("backtracks", backtracks)
        self._add_measurements("assignments", assignments)
        self._add_measurements("board_validations", board_validations)
        result_board = Board(board_manager.size_, result)
        if not board_manager.is_goal_state(result_board):
            sys.exit(1)


class BTSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "BT"


class BTMRVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(select_unassigned_var_type="mrv")

    def get_name(self):
        return "BT-MRV"


class BTLCVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(order_domain_values_type="lcv")

    def get_name(self):
        return "BT-LCV"


class BTMRVLCVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(select_unassigned_var_type="mrv", order_domain_values_type="lcv")

    def get_name(self):
        return "BT-LCV-MRV"


class FCSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_fc=True)

    def get_name(self):
        return "FC"


class FCMRVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_fc=True, select_unassigned_var_type="mrv")

    def get_name(self):
        return "FC-MRV"


class FCLCVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_fc=True, order_domain_values_type="lcv")

    def get_name(self):
        return "FC-LCV"


class FCMRVLCVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_fc=True, select_unassigned_var_type="mrv", order_domain_values_type="lcv")

    def get_name(self):
        return "FC-LCV-MRV"


class FCACSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_ac=True, use_fc=True)

    def get_name(self):
        return "AC-FC"


class FCACMRVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_ac=True, use_fc=True, select_unassigned_var_type="mrv")

    def get_name(self):
        return "AC-FC-MRV"


class FCACMRVLCVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_ac=True, use_fc=True, order_domain_values_type="lcv", select_unassigned_var_type="mrv")

    def get_name(self):
        return "AC-FC-LCV-MRV"


class ACSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_ac=True)

    def get_name(self):
        return "AC"


class ACMRVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_ac=True, select_unassigned_var_type="mrv")

    def get_name(self):
        return "AC-MRV"


class ACLCVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_ac=True, order_domain_values_type="lcv")

    def get_name(self):
        return "AC-LCV"


class ACMRVLCVSolverRunner(CSPSolverRunner):
    def __init__(self):
        super().__init__(use_ac=True, select_unassigned_var_type="mrv", order_domain_values_type="lcv")

    def get_name(self):
        return "AC-LCV-MRV"


class HillClimbingIterationsSolverRunner(SolverRunner):
    def __init__(self, iterations):
        super().__init__(HILL_CLIMBING_MEASUREMENTS)
        self.iterations = iterations

    def _solve_board(self, board_manager, timeout_seconds=None):
        problem = CalcudokuProblem(board_manager.size_, board_manager.cages_, board_manager.operations_)
        solver = HillClimbingSolver(problem, self.iterations)
        solver.solve(lambda state: self._add_measurements("conflicts", state.value))

    def get_name(self):
        return f"Hill Climbing - {self.iterations} Iterations"


class HillClimbingCompareSolverRunner(SolverRunner):
    def __init__(self, iterations):
        super().__init__(HILL_CLIMBING_MEASUREMENTS)
        self.iterations = iterations

    def _solve_board(self, board_manager, timeout_seconds=None):
        problem = CalcudokuProblem(board_manager.size_, board_manager.cages_, board_manager.operations_)
        solver = HillClimbingSolver(problem, self.iterations)
        result = solver.solve()
        self._add_measurements("conflicts", result.value)

    def get_name(self):
        return f"HC - {self.iterations} Iterations"


class GeneticSuccessRateSolverRunner(SolverRunner):
    def __init__(self, size, generations, mutation_rate, population_size):
        super().__init__(GENETIC_MEASUREMENTS)
        self.size = size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def _solve_board(self, board_manager, timeout_seconds=None):
        solver = GeneticSolver(self.size,
                               board_manager.cages_,
                               board_manager.operations_,
                               self.population_size,
                               self.generations,
                               self.mutation_rate)
        result, _, _ = solver.solve()
        self._add_measurements("success", 100 if result else 0)

    def get_name(self):
        return f"{self.size} x {self.size}"


class GeneticParameterSolverRunner(SolverRunner):
    def __init__(self, generations, mutation_rate, population_size):
        super().__init__(GENETIC_MEASUREMENTS)
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def _solve_board(self, board_manager, timeout_seconds=None):
        solver = GeneticSolver(board_manager.size_,
                               board_manager.cages_,
                               board_manager.operations_,
                               self.population_size,
                               self.generations,
                               self.mutation_rate)
        result, _, _ = solver.solve()
        self._add_measurements("success", 100 if result else 0)

    def get_name(self):
        pass


class GeneticMutationRateSolverRunner(GeneticParameterSolverRunner):
    def __init__(self, generations, mutation_rate, population_size):
        super().__init__(generations, mutation_rate, population_size)

    def get_name(self):
        return f"Genetic - Mutation Rate: {self.mutation_rate}"


class GeneticPopulationSizeSolverRunner(GeneticParameterSolverRunner):
    def __init__(self, generations, mutation_rate, population_size):
        super().__init__(generations, mutation_rate, population_size)

    def get_name(self):
        return f"Genetic - Population Size: {self.population_size}"


class GeneticBenchmarkSolverRunner(SolverRunner):
    def __init__(self, size, generations, mutation_rate, population_size):
        super().__init__(GENETIC_MEASUREMENTS)
        self.size = size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def _solve_board(self, board_manager, timeout_seconds=None):
        solver = GeneticSolver(self.size,
                               board_manager.cages_,
                               board_manager.operations_,
                               self.population_size,
                               self.generations,
                               self.mutation_rate)
        result, _, generation = solver.solve()
        self._add_measurements("success", 100 if result else 0)
        self._add_measurements("generation", generation)

    def get_name(self):
        return (f"Genetic: Size - {self.size} x {self.size}, "
                f"Generations - {self.generations}, "
                f"Mutation Rate - {self.mutation_rate}, "
                f"Population Size - {self.population_size}")


class BasicSolverRunner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def run(self, board_manager):
        pass

    def print_empty_board(self, board_manager):
        size = board_manager.size_
        cages = board_manager.cages_
        operations = board_manager.operations_
        board = [['' for _ in range(size)] for _ in range(size)]

        for cage_idx, cage in enumerate(cages):
            op, target = operations[cage_idx]
            op_symbol = self.get_operation_symbol(op)
            for (row, col) in cage:
                board[row][col] = f'{target}{op_symbol}'

        for row in board:
            print(
                ' | '.join(f'{cell:>6}' if cell else '      ' for cell in row))
            print('-' * (size * 8))

    def get_operation_symbol(self, operation):
        symbols = {
            'add': '+',
            'multiply': '×',
            'divide': '÷',
            'subtract': '-'
        }
        return symbols.get(operation, '')

    def print_solved_board(self, board_manager, assignment):
        size = board_manager.size_
        cages = board_manager.cages_
        operations = board_manager.operations_
        if assignment.shape != (size, size):
            print("Assignment size doesn't match the board size!")
            return

        board = [['' for _ in range(size)] for _ in range(size)]

        for cage_idx, cage in enumerate(cages):
            op, target = operations[cage_idx]
            op_symbol = self.get_operation_symbol(op)
            for (row, col) in cage:
                board[row][col] = f'{assignment[row][col]} ({target}{op_symbol})'

        for row in board:
            print(' | '.join(
                f'{cell:>8}' if cell else '        ' for cell in row))
            print('-' * (size * 12))


class BasicCSPSolverRunner(BasicSolverRunner):
    def __init__(self, select_unassigned_var_type,
                 order_domain_values_type, use_fc, use_ac):
        super().__init__()
        self.select_unassigned_var_type = select_unassigned_var_type
        self.order_domain_values_type = order_domain_values_type
        self.use_fc = use_fc
        self.use_ac = use_ac

    def get_name(self):
        return (f"Unassigned Variable Selection using: {self.select_unassigned_var_type}, "
                f"Order Domain Value: {self.order_domain_values_type}, "
                f"use Forward Checking: {self.use_fc}, "
                f"use Arc Consistency: {self.use_ac}")

    def run(self, board_manager):
        print(f"Solving a Calcudoku board of size {board_manager.size_}X{board_manager.size_} with Backtracking Algorithm: {self.get_name()}")
        print("Board:")
        self.print_empty_board(board_manager)
        solver = BacktrackingCSPSolver(board_manager,
                                       self.select_unassigned_var_type,
                                       self.order_domain_values_type,
                                       self.use_fc,
                                       self.use_ac)
        (result,
         backtracks_measurement,
         assignments_measurement,
         conflict_checks_measurement) = solver.solve()
        result_board = Board(board_manager.size_, result)
        if not board_manager.is_goal_state(result_board):
            print("Solution is not valid")
            sys.exit(1)

        print("Solution:")
        self.print_solved_board(board_manager, result)
        print(f"Number of Backtracks: {backtracks_measurement}")
        print(f"Number of Assignments: {assignments_measurement}")
        print(f"Number of Conflict Checks: {conflict_checks_measurement}")
        print()


class BasicHillClimbingSolverRunner(BasicSolverRunner):
    def __init__(self, iterations):
        super().__init__()
        self.iterations = iterations

    def get_name(self):
        return f"Max Iterations: {self.iterations}"

    def run(self, board_manager):
        print(f"Solving a Calcudoku board of size {board_manager.size_}X{board_manager.size_} with Hill Climbing Algorithm: {self.get_name()}")
        print("Board:")
        self.print_empty_board(board_manager)
        problem = CalcudokuProblem(board_manager.size_,
                                   board_manager.cages_,
                                   board_manager.operations_)
        solver = HillClimbingSolver(problem, self.iterations)
        final_state = solver.solve()
        board, conflicts = final_state.board, final_state.value
        print("Solution:")
        self.print_solved_board(board_manager, board.grid_)
        print(f"Solution is {'valid' if conflicts == 0 else 'invalid'}")
        print(f"Number of Conflicts: {conflicts}")
        print()


class BasicGeneticSolverRunner(BasicSolverRunner):
    def __init__(self, generations, mutation_rate, population_size):
        super().__init__()
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def get_name(self):
        return (f"Generations: {self.generations}, "
                f"Mutation Rate: {self.mutation_rate}, "
                f"Population Size: {self.population_size}")

    def run(self, board_manager):
        print(f"Solving a Calcudoku board of size {board_manager.size_}X{board_manager.size_} with Genetic Algorithm: {self.get_name()}")
        print("Board:")
        self.print_empty_board(board_manager)
        solver = GeneticSolver(board_manager.size_,
                               board_manager.cages_,
                               board_manager.operations_,
                               self.population_size,
                               self.generations,
                               self.mutation_rate)
        is_solved, result, generation = solver.solve()
        print("Solution:")
        self.print_solved_board(board_manager, np.array(result))
        print(f"Solution is {'valid' if is_solved else 'invalid'}")
        print(f"Solution Generation: {generation}")
        print()


class BasicDFSSolverRunner(BasicSolverRunner):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "DFS"

    def run(self, board_manager):
        print(f"Solving a Calcudoku board of size {board_manager.size_}X{board_manager.size_} with {self.get_name()}:")
        print("Board:")
        self.print_empty_board(board_manager)
        problem = CalcudokuBaselineProblem(board_manager)
        solver = DFSSolver(problem)
        start_time = time.time()
        result = solver.solve()
        end_time = time.time()
        result_board = Board(board_manager.size_, result.grid)
        if not board_manager.is_goal_state(result_board):
            print("Solution is not valid")
            sys.exit(1)

        print("Solution:")
        self.print_solved_board(board_manager, np.array(result.grid))
        print(f"Execution Time: {end_time - start_time} seconds")
