import numpy as np
from collections import defaultdict

from game_generator.graph import Graph


def identity(x):
    return x[0]


def divide(x):
    if max(x) % min(x) == 0:
        return int(max(x) / min(x))
    else:
        return np.nan


def subtract(x):
    return np.abs(x[1] - x[0])


OPERATIONS = {
    'divide': (2, 2, divide),
    'multiply': (2, 5, np.multiply.reduce),
    'add': (2, np.inf, np.add.reduce),
    'subtract': (2, 2, subtract),
    'none': (1, 1, identity),
}


def random_board(size):
    vals = list(range(1, size + 1))

    board = np.zeros((size, size), dtype=int)
    for row in range(size):
        while True:
            board[row, :] = 0
            for col in range(size):
                possible = set(vals) - set(list(board[row, :]) + list(board[:, col]))

                if len(possible) == 0:
                    break
                board[row, col] = np.random.choice(list(possible))
            else:
                break
    return board.flatten()


def partition_board(size, max_partition_size,
                    initial_choice_size_factor=2,
                    merge_size_factor=0.8,
                    average_size_stop=3):
    graph = Graph(size)

    average_node_size = 1

    while average_node_size < average_size_stop:
        p = np.ones(len(graph.nodes))
        for i, n in enumerate(graph.nodes):
            p[i] /= np.exp(len(n.coords))**initial_choice_size_factor
        p /= np.sum(p)
        node_idx = np.random.choice(list(range(len(graph.nodes))), p=p)

        p = np.ones(len(graph.edges[node_idx]))
        for i, n in enumerate(graph.edges[node_idx]):
            p[i] /= np.exp(len(n.coords))**merge_size_factor
        p /= np.sum(p)

        merge_node = np.random.choice(graph.edges[node_idx], p=p)
        merge_idx = graph.nodes.index(merge_node)

        if len(graph.nodes[node_idx].coords) + len(merge_node.coords) > max_partition_size:
            continue

        graph.merge(node_idx, merge_idx)

        node_lengths = [len(n.coords) for n in graph.nodes]

        average_node_size = np.average(node_lengths)

    partitions = [n.coords for n in graph.nodes]

    return partitions


def possible_operations(partitions):
    all_possibles = []
    for partition in partitions:
        possibles = []

        for name, (min_size, max_size, operation) in OPERATIONS.items():
            if min_size <= len(partition) <= max_size:
                val = operation(partition)
                if ~np.isnan(val):
                    possibles.append((name, val))
        all_possibles.append(possibles)
    return all_possibles


class Calcudoku:
    def __init__(self):
        self._groups = []
        self._board = None
        self._partitions = None
        self._operations = None

    @classmethod
    def generate(cls, size: int,
                 operation_p=None,
                 operation_uniformity_factor=2):

        result = cls()

        result._board = random_board(size)

        partitions = partition_board(size, max_partition_size=7)

        partition_values = []
        for p in partitions:
            partition_values.append([result._board[v] for v in p])

        possibles = possible_operations(partition_values)

        chosen = []
        chosen_operations = defaultdict(int)
        for possible in possibles:

            p = np.ones(len(possible))
            for i, (op, v) in enumerate(possible):
                if operation_p:
                    p[i] *= operation_p[op]
                p[i] /= np.exp(chosen_operations[(len(possible), op)])**operation_uniformity_factor

            p /= np.sum(p)

            random_index = np.random.choice(list(range(len(possible))), p=p)
            chosen.append(possible[random_index])

            chosen_operations[(len(possible), possible[random_index][0])] += 1

        result._partitions = partitions
        result._operations = chosen

        return result

    @property
    def board(self):
        return self._board

    @property
    def partitions(self):
        return self._partitions

    @property
    def operations(self):
        return self._operations
