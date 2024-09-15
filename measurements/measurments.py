import datetime
import os
import statistics
import sys
import time
from threading import Semaphore
from typing import List
import logging
import threading

from baseline.calcudoku_baseline_problem import CalcudokuBaselineProblem
from baseline.dfs_solver import DFSSolver
from calcudoku_generator import CalcudokuGenerator
from csp.csp_solver import BacktrackingCSPSolver
from game_board.game_board import BoardManager
from local_search.genetic_solver import GeneticSolver
from measurements.solver_runner import SolverRunner, MEASUREMENTS, HILL_CLIMBING_MEASUREMENTS
import matplotlib.pyplot as plt

GRAPHS_DIR_PATH = "graphs"
MAX_TIMEOUTS_IN_A_ROW = 10


def _init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    return logger


MEASUREMENTS_LOGGER = _init_logger('MEASUREMENTS_LOGGER')
MEASUREMENTS_LOGGER.disabled = False


class Measurements:
    def __init__(self, solvers, solvers_type, solvers_group):
        self.solvers: List[SolverRunner] = [solver() for solver in solvers]
        self.solvers_type = solvers_type
        self.solvers_group = solvers_group

    def start(self, size, board_amount, plot=False):
        self._run(size, board_amount)
        self._print(size, board_amount)
        if plot:
            self._plot(size, board_amount)

    def _run(self, size, board_amount):
        for solver in self.solvers:
            solver.reset_measurements()

        threads = []
        max_concurrent_threads = os.cpu_count() or 1
        semaphore = Semaphore(max_concurrent_threads)

        for i in range(board_amount):
            game = CalcudokuGenerator().generate(size)
            board_manager = BoardManager(size, game.cages, game.operations)
            for solver in self.solvers:
                semaphore.acquire()
                t = threading.Thread(target=self._run_solver, args=(solver, board_manager, semaphore, i))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

    def _plot(self, size, iteration_num):
        graphs_dir = os.path.join(os.getcwd(), 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        curr_graphs_dir_name = f'{self.solvers_type} - compare {self.solvers_group} group - size={size} - iter={iteration_num} - {timestamp}'
        curr_graphs_dir_path = os.path.join(graphs_dir, curr_graphs_dir_name)
        os.makedirs(curr_graphs_dir_path)

        for measurement in MEASUREMENTS:
            plt.figure(figsize=(10, 7))
            plt.title(
                f'{self.solvers_type} - {self.solvers_group}: Average {measurement}\nfor board size {size} and {iteration_num} iterations',
                y=1.02, fontsize=14)

            solver_names = [solver.get_name() for solver in self.solvers]
            avg_measurements = [solver.get_avg_for_measurements(measurement) for solver in self.solvers]

            bars = plt.bar(solver_names, avg_measurements)

            plt.xticks(rotation=45, ha='right')
            plt.ylabel(f'Avg {measurement}')
            plt.gcf().subplots_adjust(bottom=0.15)
            for bar, value in zip(bars, avg_measurements):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{value:.2f}', ha='center', va='bottom')

            file_name = f'{measurement}.png'
            file_path = os.path.join(curr_graphs_dir_path, file_name)
            plt.savefig(file_path)

    def _print(self, size, iteration_num):
        for solver in self.solvers:
            print(solver.get_name())
            lyx_format = f"{solver.get_name()}"
            for measurement in MEASUREMENTS:
                avg_measurement = solver.get_avg_for_measurements(measurement)
                if avg_measurement is not None:
                    lyx_format += f" & {avg_measurement}"
                    print(f'    - Average {measurement}: {avg_measurement}')
            MEASUREMENTS_LOGGER.info(lyx_format)
            print("-------------")

    def _run_solver(self, solver, board_manager, semaphore, i):
        try:
            solver.run(board_manager)
        except Exception as e:
            raise e
        finally:
            MEASUREMENTS_LOGGER.info(
                f"{datetime.datetime.fromtimestamp(time.time())}: {solver.get_name()} finished round {i}")
            semaphore.release()


class TimeMeasurements:
    def __init__(self, solver, solver_type):
        self.solver = solver()
        self.solver_type = solver_type
        self.time_measurements = None
        self.timeouts_counter = MAX_TIMEOUTS_IN_A_ROW
        self.lock = threading.Lock()

    def start(self, size, board_amount, timeout_seconds):
        self._run(size, board_amount, timeout_seconds)
        self._print(size, board_amount)

    def _run(self, size, board_amount, timeout_seconds):
        self.time_measurements = [None] * board_amount

        threads = []
        max_concurrent_threads = os.cpu_count() or 1
        semaphore = Semaphore(max_concurrent_threads)

        for i in range(board_amount):
            self.lock.acquire()
            if self.timeouts_counter <= 0:
                self.lock.release()
                break
            self.lock.release()
            game = CalcudokuGenerator().generate(size)
            board_manager = BoardManager(size, game.cages, game.operations)
            semaphore.acquire()
            t = threading.Thread(target=self._run_solver, args=(board_manager, semaphore, i, timeout_seconds))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def _print(self, size, iteration_num):
        not_none_time_measurements = [time for time in self.time_measurements if time is not None]
        print(self.solver.get_name())
        if self.timeouts_counter <= 0:
            print(f"    - {self.solver.get_name()} got {MAX_TIMEOUTS_IN_A_ROW} straight timeouts")
        else:
            print(f"    - success rate = {len(not_none_time_measurements) / len(self.time_measurements) * 100}%")
            mean_time = statistics.mean(not_none_time_measurements) if not_none_time_measurements else "N/A"
            print(f"    - Average seconds to solve board = {round(mean_time, 3)}")
        print("-------------")

    def _run_solver(self, board_manager, semaphore, i, timeout_seconds):
        start_time = time.time()
        try:
            self.lock.acquire()
            if self.timeouts_counter <= 0:
                MEASUREMENTS_LOGGER.info(f"not starting board {i}, {self.timeouts_counter} timeouts left")
                self.lock.release()
                return
            self.lock.release()
            self.solver.run(board_manager, timeout_seconds)
            self.time_measurements[i] = time.time() - start_time
            self.lock.acquire()
            self.timeouts_counter = 0 if self.timeouts_counter <= 0 else MAX_TIMEOUTS_IN_A_ROW
            self.lock.release()
            MEASUREMENTS_LOGGER.info(f"finished board {i}, {self.timeouts_counter} timeouts left")
        except TimeoutError as e:
            self.lock.acquire()
            self.timeouts_counter -= 1
            self.lock.release()
            self.time_measurements[i] = None
            MEASUREMENTS_LOGGER.info(f"timeout board {i}, {self.timeouts_counter} timeouts left")
        finally:
            semaphore.release()


class HillClimbingIterationsMeasurements:
    def __init__(self, solver):
        self.solver = solver

    def start(self, size, iterations, plot=False):
        self._run(size)
        self._print()
        if plot:
            self._plot(size, iterations)

    def _run(self, size):
        self.solver.reset_measurements()
        game = CalcudokuGenerator().generate(size)
        board_manager = BoardManager(size, game.cages, game.operations)
        self.solver.run(board_manager)

    def _plot(self, size, iterations):
        graphs_dir = os.path.join(os.getcwd(), 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        curr_graphs_dir_name = f'Hill Climbing - size={size} - {timestamp}'
        curr_graphs_dir_path = os.path.join(graphs_dir, curr_graphs_dir_name)
        os.makedirs(curr_graphs_dir_path)

        plt.figure(figsize=(10, 7))
        plt.title(f'Hill Climbing: Conflicts over Iterations\nfor board size {size} and {iterations} iterations', y=1.02, fontsize=14)
        iterations_list = list(range(1, iterations + 1))
        conflicts = self.solver.get_measurements('conflicts')
        if len(conflicts) < iterations:
            conflicts.extend([conflicts[-1]] * (iterations - len(conflicts)))
        plt.plot(iterations_list, conflicts)

        plt.xlabel('Iterations')
        plt.ylabel('Number of Conflicts')
        plt.gcf().subplots_adjust(bottom=0.15)

        file_name = 'conflicts_over_iterations.png'
        file_path = os.path.join(curr_graphs_dir_path, file_name)
        plt.savefig(file_path)

    def _print(self):
        print(self.solver.get_name())
        lyx_format = f"{self.solver.get_name()}"
        num_of_conflicts = self.solver.get_measurements('conflicts')[-1]
        if num_of_conflicts is not None:
            lyx_format += f" & {num_of_conflicts}"
            print(f'    - Num of Conflicts: {num_of_conflicts}')
        MEASUREMENTS_LOGGER.info(lyx_format)
        print("-------------")


class HillClimbingCompareMeasurements:
    def __init__(self, solvers):
        self.solvers = solvers

    def start(self, size, board_amount, plot=False):
        self._run(size, board_amount)
        self._print()
        if plot:
            self._plot(size, board_amount)

    def _run(self, size, board_amount):
        for solver in self.solvers:
            solver.reset_measurements()

        threads = []
        max_concurrent_threads = os.cpu_count() or 1
        semaphore = Semaphore(max_concurrent_threads)

        games = []
        for i in range(board_amount):
            game = CalcudokuGenerator().generate(size)
            board_manager = BoardManager(size, game.cages, game.operations)
            games.append(board_manager)

        for i in range(board_amount):
            for solver in self.solvers:
                semaphore.acquire()
                t = threading.Thread(target=self._run_solver, args=(solver, games[i], semaphore, i))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

    def _plot(self, size, board_amount):
        graphs_dir = os.path.join(os.getcwd(), 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        curr_graphs_dir_name = f'Hill Climbing - size={size} - board_amount={board_amount} - {timestamp}'
        curr_graphs_dir_path = os.path.join(graphs_dir, curr_graphs_dir_name)
        os.makedirs(curr_graphs_dir_path)

        for measurement in HILL_CLIMBING_MEASUREMENTS:
            plt.figure(figsize=(10, 7))
            plt.title(
                f'Hill Climbing: Average {measurement}\nfor board size {size} and {board_amount} boards',
                y=1.02, fontsize=14)

            solver_names = [solver.get_name() for solver in self.solvers]
            avg_measurements = [solver.get_avg_for_measurements(measurement) for solver in self.solvers]

            plt.bar(solver_names, avg_measurements)

            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Iterations')
            plt.ylabel(f'Avg {measurement}')
            plt.gcf().subplots_adjust(bottom=0.15)

            file_name = f'{measurement}.png'
            file_path = os.path.join(curr_graphs_dir_path, file_name)
            plt.savefig(file_path)

    def _print(self):
        for solver in self.solvers:
            print(solver.get_name())
            lyx_format = f"{solver.get_name()}"
            for measurement in HILL_CLIMBING_MEASUREMENTS:
                avg_measurement = solver.get_avg_for_measurements(measurement)
                if avg_measurement is not None:
                    lyx_format += f" & {avg_measurement}"
                    print(f'    - Average {measurement}: {avg_measurement}')
            MEASUREMENTS_LOGGER.info(lyx_format)
            print("-------------")

    def _run_solver(self, solver, board_manager, semaphore, i):
        try:
            solver.run(board_manager)
        except Exception as e:
            raise e
        finally:
            MEASUREMENTS_LOGGER.info(
                f"{datetime.datetime.fromtimestamp(time.time())}: {solver.get_name()} finished round {i}")
            semaphore.release()


class GeneticSuccessRateMeasurements:
    def __init__(self, solvers):
        self.solvers = solvers

    def start(self, max_size, board_amount, generations,
              mutation_rate, population_size, plot=False):
        self._run(max_size, board_amount)
        self._print()
        if plot:
            self._plot(max_size, board_amount, generations,
                       mutation_rate, population_size)

    def _run(self, max_size, board_amount):
        for size in range(3, max_size + 1):
            solver = self.solvers[size - 3]
            solver.reset_measurements()

            threads = []
            max_concurrent_threads = os.cpu_count() or 1
            semaphore = Semaphore(max_concurrent_threads)

            games = []
            for i in range(board_amount):
                game = CalcudokuGenerator().generate(size)
                board_manager = BoardManager(size, game.cages, game.operations)
                games.append(board_manager)

            for i in range(board_amount):
                semaphore.acquire()
                t = threading.Thread(target=self._run_solver, args=(solver, games[i], semaphore, i))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

    def _plot(self, size, board_amount, generations,
              mutation_rate, population_size):
        graphs_dir = os.path.join(os.getcwd(), 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        curr_graphs_dir_name = f'Genetic - success-rate - max-size={size} - {timestamp}'
        curr_graphs_dir_path = os.path.join(graphs_dir, curr_graphs_dir_name)
        os.makedirs(curr_graphs_dir_path)

        plt.figure(figsize=(10, 7))
        plt.title(f'Genetic: Success Rate for {board_amount} boards with '
                  f'different board sizes,\n '
                  f'{generations} generations, {mutation_rate} mutation rate and '
                  f'{population_size} population size', y=1.02, fontsize=14)

        solver_names = [solver.get_name() for solver in self.solvers]
        avg_measurements = [solver.get_avg_for_measurements("success") for solver in self.solvers]

        plt.bar(solver_names, avg_measurements)

        plt.xlabel('Board Size')
        plt.ylabel('Success Rate (%)')
        plt.gcf().subplots_adjust(bottom=0.15)

        file_name = 'genetic_success_rate.png'
        file_path = os.path.join(curr_graphs_dir_path, file_name)
        plt.savefig(file_path)

    def _print(self):
        for solver in self.solvers:
            print(solver.get_name())
            lyx_format = f"{solver.get_name()}"
            avg_measurement = solver.get_avg_for_measurements("success")
            if avg_measurement is not None:
                lyx_format += f" & {avg_measurement}"
                print(f'    - Average success rate: {avg_measurement}')
            MEASUREMENTS_LOGGER.info(lyx_format)
            print("-------------")

    def _run_solver(self, solver, board_manager, semaphore, i):
        try:
            solver.run(board_manager)
        except Exception as e:
            raise e
        finally:
            MEASUREMENTS_LOGGER.info(
                f"{datetime.datetime.fromtimestamp(time.time())}: {solver.get_name()} finished round {i}")
            semaphore.release()


class GeneticMutationRateMeasurements:
    def __init__(self, solvers):
        self.solvers = solvers

    def start(self, size, board_amount, generations, population_size, plot=False):
        self._run(size, board_amount)
        self._print()
        if plot:
            self._plot(size, board_amount, generations, population_size)

    def _run(self, size, board_amount):
        for solver in self.solvers:
            solver.reset_measurements()

        threads = []
        max_concurrent_threads = os.cpu_count() or 1
        semaphore = Semaphore(max_concurrent_threads)

        games = []
        for i in range(board_amount):
            game = CalcudokuGenerator().generate(size)
            board_manager = BoardManager(size, game.cages, game.operations)
            games.append(board_manager)

        for i in range(board_amount):
            for solver in self.solvers:
                semaphore.acquire()
                t = threading.Thread(target=self._run_solver, args=(solver, games[i], semaphore, i))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

    def _plot(self, size, board_amount, generations, population_size):
        graphs_dir = os.path.join(os.getcwd(), 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        curr_graphs_dir_name = f'Genetic - mutation-rate - size={size} - {timestamp}'
        curr_graphs_dir_path = os.path.join(graphs_dir, curr_graphs_dir_name)
        os.makedirs(curr_graphs_dir_path)

        plt.figure(figsize=(10, 7))
        plt.title(f'Genetic: Success Rate over Mutation Rate for {board_amount} boards of size '
                  f'{size} x {size},\n{generations} generations, and {population_size} population size',
                  y=1.02, fontsize=14)

        mutation_rates_list = [solver.mutation_rate for solver in self.solvers]
        avg_measurements = [solver.get_avg_for_measurements("success") for solver in self.solvers]

        plt.plot(mutation_rates_list, avg_measurements)

        plt.xlabel('Mutation Rate')
        plt.ylabel('Success Rate (%)')
        plt.gcf().subplots_adjust(bottom=0.15)

        file_name = 'genetic_success_rate_mutation_rate.png'
        file_path = os.path.join(curr_graphs_dir_path, file_name)
        plt.savefig(file_path)

    def _print(self):
        for solver in self.solvers:
            print(solver.get_name())
            lyx_format = f"{solver.get_name()}"
            avg_measurement = solver.get_avg_for_measurements("success")
            if avg_measurement is not None:
                lyx_format += f" & {avg_measurement}"
                print(f'    - Average success rate: {avg_measurement}')
            MEASUREMENTS_LOGGER.info(lyx_format)
            print("-------------")

    def _run_solver(self, solver, board_manager, semaphore, i):
        try:
            solver.run(board_manager)
        except Exception as e:
            raise e
        finally:
            MEASUREMENTS_LOGGER.info(
                f"{datetime.datetime.fromtimestamp(time.time())}: {solver.get_name()} finished round {i}")
            semaphore.release()


class GeneticPopulationSizeMeasurements:
    def __init__(self, solvers):
        self.solvers = solvers

    def start(self, size, board_amount, mutation_rate, generations, plot=False):
        self._run(size, board_amount)
        self._print()
        if plot:
            self._plot(size, board_amount, generations, mutation_rate)

    def _run(self, size, board_amount):
        for solver in self.solvers:
            solver.reset_measurements()

        threads = []
        max_concurrent_threads = os.cpu_count() or 1
        semaphore = Semaphore(max_concurrent_threads)

        games = []
        for i in range(board_amount):
            game = CalcudokuGenerator().generate(size)
            board_manager = BoardManager(size, game.cages, game.operations)
            games.append(board_manager)

        for i in range(board_amount):
            for solver in self.solvers:
                semaphore.acquire()
                t = threading.Thread(target=self._run_solver, args=(solver, games[i], semaphore, i))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

    def _plot(self, size, board_amount, generations, mutation_rate):
        graphs_dir = os.path.join(os.getcwd(), 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        curr_graphs_dir_name = f'Genetic - population-size - size={size} - {timestamp}'
        curr_graphs_dir_path = os.path.join(graphs_dir, curr_graphs_dir_name)
        os.makedirs(curr_graphs_dir_path)

        plt.figure(figsize=(10, 7))
        plt.title(f'Genetic: Success Rate over Population Size for {board_amount} boards of size '
                  f'{size} x {size},\n{generations} generations, and {mutation_rate} mutation rate',
                  y=1.02, fontsize=14)

        population_size_list = [str(solver.population_size) for solver in self.solvers]
        avg_measurements = [solver.get_avg_for_measurements("success") for solver in self.solvers]

        plt.bar(population_size_list, avg_measurements)
        plt.xticks(population_size_list)
        plt.xlabel('Population Size')
        plt.ylabel('Success Rate (%)')
        plt.gcf().subplots_adjust(bottom=0.15)

        file_name = 'genetic_success_rate_population_size.png'
        file_path = os.path.join(curr_graphs_dir_path, file_name)
        plt.savefig(file_path)

    def _print(self):
        for solver in self.solvers:
            print(solver.get_name())
            lyx_format = f"{solver.get_name()}"
            avg_measurement = solver.get_avg_for_measurements("success")
            if avg_measurement is not None:
                lyx_format += f" & {avg_measurement}"
                print(f'    - Average success rate: {avg_measurement}')
            MEASUREMENTS_LOGGER.info(lyx_format)
            print("-------------")

    def _run_solver(self, solver, board_manager, semaphore, i):
        try:
            solver.run(board_manager)
        except Exception as e:
            raise e
        finally:
            MEASUREMENTS_LOGGER.info(
                f"{datetime.datetime.fromtimestamp(time.time())}: {solver.get_name()} finished round {i}")
            semaphore.release()


class GeneticBenchmarkMeasurements:
    def __init__(self, solver):
        self.solver = solver

    def start(self, size, board_amount):
        self._run(size, board_amount)
        self._print()

    def _run(self, size, board_amount):
        self.solver.reset_measurements()

        threads = []
        max_concurrent_threads = os.cpu_count() or 1
        semaphore = Semaphore(max_concurrent_threads)

        games = []
        for i in range(board_amount):
            game = CalcudokuGenerator().generate(size)
            board_manager = BoardManager(size, game.cages, game.operations)
            games.append(board_manager)

        for i in range(board_amount):
            semaphore.acquire()
            t = threading.Thread(target=self._run_solver, args=(self.solver, games[i], semaphore, i))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def _print(self):
        print(self.solver.get_name())
        lyx_format = f"{self.solver.get_name()}"
        avg_success_rate = self.solver.get_avg_for_measurements("success")
        avg_generation = self.solver.get_avg_for_measurements("generation")
        if avg_success_rate is not None and avg_generation is not None:
            lyx_format += f" & {avg_success_rate} & {avg_generation}"
            print(f'    - Average success rate: {avg_success_rate}, Average generation: {avg_generation}')
        MEASUREMENTS_LOGGER.info(lyx_format)
        print("-------------")

    def _run_solver(self, solver, board_manager, semaphore, i):
        try:
            solver.run(board_manager)
        except Exception as e:
            raise e
        finally:
            MEASUREMENTS_LOGGER.info(
                f"{datetime.datetime.fromtimestamp(time.time())}: {solver.get_name()} finished round {i}")
            semaphore.release()


class BaselineMeasurements:
    def __init__(self, size, amount):
        self.size = size
        self.amount = amount
        self.boards = []
        for _ in range(self.amount):
            game = CalcudokuGenerator.generate(self.size)
            board, cages, operations = game.board, game.cages, game.operations
            board_manager = BoardManager(self.size, cages, operations)
            self.boards.append(board_manager)
        self.dfs_times = []
        self.ac_mrv_times = []
        self.genetic_times = []

    def start(self, plot, population_size, generations, mutation_rate, timeout):
        self._run(population_size, generations, mutation_rate, timeout)
        self._print()
        if plot:
            self._plot(timeout)

    def _run(self, population_size, generations, mutation_rate, timeout):
        dfs_timeouts, ac_mrv_timeouts, genetic_timeouts = 0, 0, 0
        should_run_dfs, should_run_ac_mrv, should_run_genetic = True, True, True

        for i in range(self.amount):
            if should_run_dfs:
                dfs_time = self.measure_time(
                    DFSSolver(CalcudokuBaselineProblem(self.boards[i])),
                    "DFS",
                    i,
                    timeout)
                self.dfs_times.append(dfs_time)
                if not dfs_time:
                    dfs_timeouts += 1
                else:
                    dfs_timeouts = 0
                if dfs_timeouts >= MAX_TIMEOUTS_IN_A_ROW:
                    should_run_dfs = False

            if should_run_ac_mrv:
                ac_mrv_time = self.measure_time(
                    BacktrackingCSPSolver(self.boards[i], "mrv", "random", False,
                                          True),
                    "AC-MRV",
                    i,
                    timeout)
                self.ac_mrv_times.append(ac_mrv_time)
                if not ac_mrv_time:
                    ac_mrv_timeouts += 1
                else:
                    ac_mrv_timeouts = 0
                if ac_mrv_timeouts >= MAX_TIMEOUTS_IN_A_ROW:
                    should_run_ac_mrv = False

            if should_run_genetic:
                genetic_time = self.measure_time(
                    GeneticSolver(self.boards[i].size_, self.boards[i].cages_,
                                  self.boards[i].operations_, population_size,
                                  generations, mutation_rate),
                    "Genetic",
                    i,
                    timeout)
                self.genetic_times.append(genetic_time)
                if not genetic_time:
                    genetic_timeouts += 1
                else:
                    genetic_timeouts = 0
                if genetic_timeouts >= MAX_TIMEOUTS_IN_A_ROW:
                    should_run_genetic = False

    def _print(self):
        dfs_reached_timeout = all(time is None for time in self.dfs_times)
        ac_mrv_reached_timeout = all(time is None for time in self.ac_mrv_times)
        genetic_reached_timeout = all(time is None for time in self.genetic_times)

        print("DFS:")
        print(f"    - Average time: {statistics.mean([time for time in self.dfs_times if time is not None]) if not dfs_reached_timeout else 'N/A'}")
        print(f"    - Timeouts: {len([time for time in self.dfs_times if time is None]) if not dfs_reached_timeout else 'Reached Max Timeouts'}")
        print("AC MRV:")
        print(f"    - Average time: {statistics.mean([time for time in self.ac_mrv_times if time is not None]) if not ac_mrv_reached_timeout else 'N/A'}")
        print(f"    - Timeouts: {len([time for time in self.ac_mrv_times if time is None]) if not ac_mrv_reached_timeout else 'Reached Max Timeouts'}")
        print("Genetic:")
        print(f"    - Average time: {statistics.mean([time for time in self.genetic_times if time is not None]) if not genetic_reached_timeout else 'N/A'}")
        print(f"    - Timeouts: {len([time for time in self.genetic_times if time is None]) if not genetic_reached_timeout else 'Reached Max Timeouts'}")
        print("-------------")

    def _plot(self, timeout):
        graphs_dir = os.path.join(os.getcwd(), 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        curr_graphs_dir_name = f'Baseline - compare - size={self.size} - {timestamp}'
        curr_graphs_dir_path = os.path.join(graphs_dir, curr_graphs_dir_name)
        os.makedirs(curr_graphs_dir_path)

        plt.figure(figsize=(10, 7))
        plt.title(
            f'Baseline: Average Time for {self.amount} boards of size {self.size}X{self.size}\nTimeout: {timeout} seconds',
            y=1.02, fontsize=14)

        dfs_reached_timeout = all(time is None for time in self.dfs_times)
        ac_mrv_reached_timeout = all(time is None for time in self.ac_mrv_times)
        genetic_reached_timeout = all(time is None for time in self.genetic_times)

        solvers_names_list = ["DFS - Timeout" if dfs_reached_timeout else "DFS",
                              "AC MRV - Timeout" if ac_mrv_reached_timeout else "AC MRV",
                              "Genetic - Timeout" if genetic_reached_timeout else "Genetic"]
        avg_measurements = [
            statistics.mean([time for time in self.dfs_times if time is not None]) if not dfs_reached_timeout else 0,
            statistics.mean([time for time in self.ac_mrv_times if time is not None]) if not ac_mrv_reached_timeout else 0,
            statistics.mean([time for time in self.genetic_times if time is not None] if not genetic_reached_timeout else 0)
        ]

        bars = plt.bar(solvers_names_list, avg_measurements)

        plt.ylabel('Average Time (sec)')
        plt.gcf().subplots_adjust(bottom=0.15)
        for bar, value in zip(bars, avg_measurements):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value:.6f}', ha='center', va='bottom')

        file_name = 'baseline_time_compare.png'
        file_path = os.path.join(curr_graphs_dir_path, file_name)
        plt.savefig(file_path)

    def measure_time(self, solver, solver_name, iteration, timeout):
        start_time = time.time()
        try:
            print(f"starting {solver_name}, round {iteration}")
            solver.solve(timeout)
        except TimeoutError:
            print(f"timeout {solver_name}, round {iteration}")
            return None
        end_time = time.time()
        print(f"finished {solver_name}, round {iteration}")
        return end_time - start_time
