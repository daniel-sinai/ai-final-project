import argparse
from measurements.measurments import *
from measurements.solver_runner import *


CSP_SOLVERS_7_AND_BELOW = [ACMRVLCVSolverRunner, FCACMRVLCVSolverRunner, FCACMRVSolverRunner]
CSP_SOLVERS_6_AND_BELOW = [FCACSolverRunner, ACSolverRunner, ACMRVSolverRunner,
                           ACLCVSolverRunner] + CSP_SOLVERS_7_AND_BELOW
CSP_SOLVERS_5_AND_BELOW = [FCMRVSolverRunner] + CSP_SOLVERS_6_AND_BELOW
CSP_SOLVERS_4_AND_BELOW = ([FCSolverRunner, FCLCVSolverRunner, FCMRVLCVSolverRunner] + CSP_SOLVERS_5_AND_BELOW)
CSP_SOLVERS_3_AND_BELOW = ([BTSolverRunner, BTMRVSolverRunner, BTLCVSolverRunner, BTMRVLCVSolverRunner] +
                           CSP_SOLVERS_4_AND_BELOW)


SOLVERS_GROUPS = {
    "csp": {
        "all": None,
        "3&below": CSP_SOLVERS_3_AND_BELOW,
        "4&below": CSP_SOLVERS_4_AND_BELOW,
        "5&below": CSP_SOLVERS_5_AND_BELOW,
        "6&below": CSP_SOLVERS_6_AND_BELOW,
        "7&below": CSP_SOLVERS_7_AND_BELOW,
    }
}

NAME_TO_SOLVER = {
    "bt": BTSolverRunner,
    "bt-mrv": BTMRVSolverRunner,
    "bt-lcv": BTLCVSolverRunner,
    "bt-mrv-lcv": BTMRVLCVSolverRunner,
    "fc": FCSolverRunner,
    "fc-mrv": FCMRVSolverRunner,
    "fc-lcv": FCLCVSolverRunner,
    "fc-lcv-mrv": FCMRVLCVSolverRunner,
    "ac-fc": FCACSolverRunner,
    "ac-fc-mrv": FCACMRVSolverRunner,
    "ac-fc-lcv-mrv": FCACMRVLCVSolverRunner,
    "ac": ACSolverRunner,
    "ac-mrv": ACMRVSolverRunner,
    "ac-lcv": ACLCVSolverRunner,
    "ac-lcv-mrv": ACMRVLCVSolverRunner
}


def pretty_print_headline(headline):
    headline = f"*** {headline} ***"
    asterisks = "*" * len(headline)
    print(asterisks)
    print(headline)
    print(asterisks)


def handle_csp_compare(args):
    run_type_msg = f"""Running csp comparison for "{args.group}" solver group with {args.amount} {args.size}X{args.size} generated boards"""
    pretty_print_headline(run_type_msg)
    measurements = Measurements(SOLVERS_GROUPS["csp"][args.group], args.command, args.group)
    measurements.start(args.size, args.amount, plot=args.graphs)


def handle_csp_time_measurement(args):
    run_type_msg = f"""Running csp time measurement for "{args.solver}" solver with {args.amount} {args.size}X{args.size} generated boards with {args.timeout} seconds timeout"""
    pretty_print_headline(run_type_msg)
    measurements = TimeMeasurements(NAME_TO_SOLVER[args.solver], args.command)
    measurements.start(args.size, args.amount, args.timeout)


def handle_hill_climbing_iterations(args):
    run_type_msg = f"""Running Hill Climbing solver with a board of size {args.size}X{args.size} for {args.iterations} iterations"""
    pretty_print_headline(run_type_msg)
    measurements = HillClimbingIterationsMeasurements(HillClimbingIterationsSolverRunner(args.iterations))
    measurements.start(args.size, args.iterations, args.graphs)


def handle_hill_climbing_compare(args):
    run_type_msg = f"""Running Hill Climbing comparison for {args.amount} boards of size {args.size}X{args.size} with maximum {args.iterations} iterations"""
    pretty_print_headline(run_type_msg)
    measurements = HillClimbingCompareMeasurements([HillClimbingCompareSolverRunner(iters) for iters in range(args.increment, args.iterations + 1, args.increment)])
    measurements.start(args.size, args.amount, plot=args.graphs)


def handle_genetic_success_rate(args):
    run_type_msg = f"""Running Genetic solver with: {args.amount} boards of 
    max size {args.max_size}X{args.max_size}, {args.mutation_rate} mutation rate, 
    {args.population_size} population size, for {args.generations} generations"""

    pretty_print_headline(run_type_msg)
    measurements = GeneticSuccessRateMeasurements(
        [GeneticSuccessRateSolverRunner(
            size,
            args.generations,
            args.mutation_rate,
            args.population_size
        ) for size in range(3, args.max_size + 1)]
    )
    measurements.start(args.max_size, args.amount, args.generations,
                       args.mutation_rate, args.population_size,
                       args.graphs)


def handle_genetic_mutation_rate_comparison(args):
    run_type_msg = f"""Running Genetic solver with: {args.amount} boards of
    size {args.size}X{args.size}, multiple mutation rates between 0 and 1,
    {args.population_size} population size, for {args.generations} generations"""

    pretty_print_headline(run_type_msg)
    measurements = GeneticMutationRateMeasurements(
        [GeneticMutationRateSolverRunner(
            args.generations,
            rate,
            args.population_size
        ) for rate in np.linspace(0, 1, 11)]
    )
    measurements.start(args.size, args.amount, args.generations,
                       args.population_size, args.graphs)


def handle_genetic_population_size_comparison(args):
    run_type_msg = f"""Running Genetic solver with: {args.amount} boards of
        size {args.size}X{args.size}, multiple population sizes between {args.increment} and {args.max_population_size},
        {args.mutation_rate} mutation rate, for {args.generations} generations"""

    pretty_print_headline(run_type_msg)
    measurements = GeneticPopulationSizeMeasurements(
        [GeneticPopulationSizeSolverRunner(
            args.generations,
            args.mutation_rate,
            population_size
        ) for population_size in range(args.increment, args.max_population_size + 1, args.increment)]
    )
    measurements.start(args.size, args.amount, args.mutation_rate,
                       args.generations, args.graphs)


def handle_genetic_benchmark(args):
    run_type_msg = f"""Running benchmark for Genetic solver with: {args.amount} boards of
            size {args.size}X{args.size}, {args.population_size} population size,
            {args.mutation_rate} mutation rate, for maximum {args.max_generations} generations"""
    pretty_print_headline(run_type_msg)
    measurements = GeneticBenchmarkMeasurements(
        GeneticBenchmarkSolverRunner(
            args.size,
            args.max_generations,
            args.mutation_rate,
            args.population_size
        )
    )
    measurements.start(args.size, args.amount)


def handle_basic_csp(args):
    run_type_msg = f"""Running csp solver on a {args.size}X{args.size} board"""
    pretty_print_headline(run_type_msg)
    game = CalcudokuGenerator().generate(args.size)
    board_manager = BoardManager(args.size, game.cages, game.operations)
    runner = BasicCSPSolverRunner(args.select_unassigned_var,
                                  args.order_domain_values,
                                  args.use_forward_checking,
                                  args.use_arc_consistency)
    runner.run(board_manager)


def handle_basic_hill_climbing(args):
    run_type_msg = f"""Running Hill Climbing solver on a {args.size}X{args.size} board"""
    pretty_print_headline(run_type_msg)
    game = CalcudokuGenerator().generate(args.size)
    board_manager = BoardManager(args.size, game.cages, game.operations)
    runner = BasicHillClimbingSolverRunner(args.iterations)
    runner.run(board_manager)


def handle_basic_genetic(args):
    run_type_msg = f"""Running Genetic solver on a {args.size}X{args.size} board"""
    pretty_print_headline(run_type_msg)
    game = CalcudokuGenerator().generate(args.size)
    board_manager = BoardManager(args.size, game.cages, game.operations)
    runner = BasicGeneticSolverRunner(args.generations, args.mutation_rate, args.population_size)
    runner.run(board_manager)


def handle_basic_dfs(args):
    run_type_msg = f"""Running DFS solver on a {args.size}X{args.size} board"""
    pretty_print_headline(run_type_msg)
    game = CalcudokuGenerator().generate(args.size)
    board_manager = BoardManager(args.size, game.cages, game.operations)
    runner = BasicDFSSolverRunner()
    runner.run(board_manager)


def handle_baseline_compare(args):
    run_type_msg = f"""Running time comparison of AC-MRV and Genetic against DFS Baseline 
    with {args.amount} {args.size}X{args.size} generated boards"""
    pretty_print_headline(run_type_msg)
    measurements = BaselineMeasurements(args.size, args.amount)
    measurements.start(args.graphs, args.population_size, args.generations, args.mutation_rate, args.timeout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the application')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # csp command
    csp_parser = subparsers.add_parser('csp', help='Commands to run the csp solvers')

    # Subcommands under csp command
    csp_subparsers = csp_parser.add_subparsers(dest='subcommand', required=True)

    # Compare subcommand
    compare_parser = csp_subparsers.add_parser('compare', help='Commands to compare the csp solvers')

    # Flags for compare subcommand
    compare_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku board')
    compare_parser.add_argument('--amount', type=int, default=10, help='Amount of boards to generate for comparison')
    compare_parser.add_argument('--group', type=str, default="3&below", help='Group of solvers to participate in the comparison')
    compare_parser.add_argument('--graphs', action='store_true', help='Whether to generate graphs or not')

    # time-measurement subcommand
    time_measurement_parser = csp_subparsers.add_parser('time-measurement', help='Commands to measure the time it takes a solver to solve board')

    # Flags for time-measurement subcommand
    time_measurement_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku board')
    time_measurement_parser.add_argument('--amount', type=int, default=10, help='Amount of boards to generate for comparison')
    time_measurement_parser.add_argument('--solver', type=str, default="bt", choices=NAME_TO_SOLVER.keys(), help='Name of the solver to use')
    time_measurement_parser.add_argument('--timeout', type=int, default=30)

    # hill climbing command
    hill_climbing_parser = subparsers.add_parser('hill-climbing', help='Commands to run the hill climbing solver')

    # Subcommands under hill climbing command
    hill_climbing_subparsers = hill_climbing_parser.add_subparsers(dest='subcommand', required=True)

    # Iterations subcommand
    iterations_parser = hill_climbing_subparsers.add_parser('iterations', help='Commands to run the hill climbing solver with a fixed amount of iterations on a single board')

    # Flags for iterations subcommand
    iterations_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku board')
    iterations_parser.add_argument('--iterations', type=int, default=20, help='Max iterations for the hill climbing solver')
    iterations_parser.add_argument('--graphs', action='store_true', help='Whether to generate graphs or not')

    # Compare subcommand
    hill_climbing_compare_parser = hill_climbing_subparsers.add_parser('compare', help='Commands to compare the hill climbing solver on multiple boards with different initial states and iterations')

    # Flags for compare subcommand
    hill_climbing_compare_parser.add_argument('--size', type=int, default=6, help='Size of the Calcudoku boards')
    hill_climbing_compare_parser.add_argument('--amount', type=int, default=30, help='Amount of boards to generate for comparison')
    hill_climbing_compare_parser.add_argument('--iterations', type=int, default=15, help='Max iterations for the hill climbing solver')
    hill_climbing_compare_parser.add_argument('--increment', type=int, default=3, help='The increment of the iterations for the hill climbing solver')
    hill_climbing_compare_parser.add_argument('--graphs', action='store_true', help='Whether to generate graphs or not')

    # genetic command
    genetic_parser = subparsers.add_parser('genetic', help='Commands to run the genetic solver')

    # Subcommands under genetic command
    genetic_subparsers = genetic_parser.add_subparsers(dest='subcommand', required=True)

    # Success rate subcommand
    success_rate_parser = genetic_subparsers.add_parser('success-rate', help='Commands to run the genetic solver with a fixed amount of generations on multiple boards and compare the success rate for different boards sizes')

    # Flags for success rate subcommand
    success_rate_parser.add_argument('--max-size', type=int, default=5, help='Max size of the Calcudoku boards')
    success_rate_parser.add_argument('--amount', type=int, default=50, help='Amount of boards to generate for comparison')
    success_rate_parser.add_argument('--generations', type=int, default=200, help='Max generations for the genetic solver')
    success_rate_parser.add_argument('--mutation-rate', type=float, default=0.5, help='The mutation rate for the genetic solver')
    success_rate_parser.add_argument('--population-size', type=int, default=1000, help='The population size for the genetic solver')
    success_rate_parser.add_argument('--graphs', action='store_true', help='Whether to generate graphs or not')

    # Mutation rate subcommand
    mutation_rate_parser = genetic_subparsers.add_parser('mutation-rate', help='Commands to run the genetic solver with a fixed amount of generations on multiple boards and compare the success rate for different mutation rates')

    # Flags for mutation rate subcommand
    mutation_rate_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku boards')
    mutation_rate_parser.add_argument('--amount', type=int, default=50, help='Amount of boards to generate for comparison')
    mutation_rate_parser.add_argument('--generations', type=int, default=50, help='Max generations for the genetic solver')
    mutation_rate_parser.add_argument('--population-size', type=int, default=10, help='The population size for the genetic solver')
    mutation_rate_parser.add_argument('--graphs', action='store_true', help='Whether to generate graphs or not')

    # Population size subcommand
    population_size_parser = genetic_subparsers.add_parser('population-size', help='Commands to run the genetic solver with a fixed amount of generations on multiple boards and compare the success rate for different population sizes')

    # Flags for population size subcommand
    population_size_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku boards')
    population_size_parser.add_argument('--amount', type=int, default=50, help='Amount of boards to generate for comparison')
    population_size_parser.add_argument('--generations', type=int, default=50, help='Max generations for the genetic solver')
    population_size_parser.add_argument('--max-population-size', type=int, default=10, help='The maximum population size for the genetic solver')
    population_size_parser.add_argument('--increment', type=int, default=2, help='The increment in the population size between different runs for the genetic solver')
    population_size_parser.add_argument('--mutation-rate', type=float, default=0.5, help='The mutation rate for the genetic solver')
    population_size_parser.add_argument('--graphs', action='store_true', help='Whether to generate graphs or not')

    # Benchmark subcommand
    benchmark_parser = genetic_subparsers.add_parser('benchmark', help='Commands to run the genetic solver with a fixed amount of generations on multiple boards of the same size and compare the success rate and solution generation')

    # Flags for benchmark subcommand
    benchmark_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku boards')
    benchmark_parser.add_argument('--amount', type=int, default=100, help='Amount of boards to generate for comparison')
    benchmark_parser.add_argument('--max-generations', type=int, default=50, help='Max generations for the genetic solver')
    benchmark_parser.add_argument('--population-size', type=int, default=5, help='The population size for the genetic solver')
    benchmark_parser.add_argument('--mutation-rate', type=float, default=0.3, help='The mutation rate for the genetic solver')

    # Basic Solve command
    solve_parser = subparsers.add_parser('solve', help='Commands to run a solver on a single Calcudoku board of a given size')

    # Subcommands under Basic Solve command
    solve_subparsers = solve_parser.add_subparsers(dest='subcommand', required=True)

    # Basic DFS subcommand
    dfs_solve_parser = solve_subparsers.add_parser('dfs', help='Commands to run a DFS solver on a single Calcudoku board of a given size')

    # Flags for basic DFS subcommand
    dfs_solve_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku board')

    # Basic csp subcommand
    csp_solve_parser = solve_subparsers.add_parser('csp', help='Commands to run a csp solver on a single Calcudoku board of a given size')

    # Flags for basic csp subcommand
    csp_solve_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku board')
    csp_solve_parser.add_argument('--use-forward-checking', action='store_true', help='Whether to use forward checking or not')
    csp_solve_parser.add_argument('--use-arc-consistency', action='store_true', help='Whether to use arc-consistency or not')
    csp_solve_parser.add_argument('--select-unassigned-var', type=str, default='none', help='The method to use for selecting unassigned variables')
    csp_solve_parser.add_argument('--order-domain-values', type=str, default='random', help='The method to use for ordering domain values')

    # Basic Hill Climbing subcommand
    hill_climbing_solve_parser = solve_subparsers.add_parser('hill-climbing', help='Commands to run a Hill Climbing solver on a single Calcudoku board of a given size')

    # Flags for basic Hill Climbing subcommand
    hill_climbing_solve_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku board')
    hill_climbing_solve_parser.add_argument('--iterations', type=int, default=50, help='Max iterations for the hill climbing solver')

    # Basic Genetic subcommand
    genetic_solve_parser = solve_subparsers.add_parser('genetic', help='Commands to run a Genetic solver on a single Calcudoku board of a given size')

    # Flags for basic Genetic subcommand
    genetic_solve_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku board')
    genetic_solve_parser.add_argument('--generations', type=int, default=50, help='Max generations for the genetic solver')
    genetic_solve_parser.add_argument('--mutation-rate', type=float, default=0.3, help='The mutation rate for the genetic solver')
    genetic_solve_parser.add_argument('--population-size', type=int, default=5, help='The population size for the genetic solver')

    # Baseline compare command
    baseline_compare_parser = subparsers.add_parser('baseline-compare', help='Commands to run a comparison against DFS baseline on multiple Calcudoku boards of a given size')

    # Flags for Baseline compare command
    baseline_compare_parser.add_argument('--size', type=int, default=3, help='Size of the Calcudoku boards')
    baseline_compare_parser.add_argument('--amount', type=int, default=100, help='Amount of boards to generate for comparison')
    baseline_compare_parser.add_argument('--population-size', type=int, default=10, help='The population size for the genetic solver')
    baseline_compare_parser.add_argument('--generations', type=int, default=50, help='Max generations for the genetic solver')
    baseline_compare_parser.add_argument('--mutation-rate', type=float, default=0.5, help='The mutation rate for the genetic solver')
    baseline_compare_parser.add_argument('--graphs', action='store_true', help='Whether to generate graphs or not')
    baseline_compare_parser.add_argument('--timeout', type=int, default=10, help='The timeout in seconds for the solvers')

    program_args = parser.parse_args()

    if program_args.command == 'csp':
        if program_args.subcommand == 'compare':
            handle_csp_compare(program_args)
        elif program_args.subcommand == 'time-measurement':
            handle_csp_time_measurement(program_args)
    elif program_args.command == 'hill-climbing':
        if program_args.subcommand == 'iterations':
            handle_hill_climbing_iterations(program_args)
        elif program_args.subcommand == 'compare':
            handle_hill_climbing_compare(program_args)
    elif program_args.command == 'genetic':
        if program_args.subcommand == 'success-rate':
            handle_genetic_success_rate(program_args)
        elif program_args.subcommand == 'mutation-rate':
            handle_genetic_mutation_rate_comparison(program_args)
        elif program_args.subcommand == 'population-size':
            handle_genetic_population_size_comparison(program_args)
        elif program_args.subcommand == 'benchmark':
            handle_genetic_benchmark(program_args)
    elif program_args.command == 'solve':
        if program_args.subcommand == 'csp':
            handle_basic_csp(program_args)
        if program_args.subcommand == 'hill-climbing':
            handle_basic_hill_climbing(program_args)
        if program_args.subcommand == 'genetic':
            handle_basic_genetic(program_args)
        if program_args.subcommand == 'dfs':
            handle_basic_dfs(program_args)
    elif program_args.command == 'baseline-compare':
        handle_baseline_compare(program_args)
