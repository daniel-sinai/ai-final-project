import random
import numpy as np


class GeneticSolver:
    def __init__(self, size, cages, operations,
                 population_size, generations, mutation_rate):
        self.GENES = range(1, size+1)
        self.size = size
        self.cages = cages
        self.operations = operations
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = [self.generate_individual() for _ in range(population_size)]

    def generate_individual(self):
        individual = []
        for _ in range(self.size):
            row = list(self.GENES)
            random.shuffle(row)
            individual.append(row)
        return individual

    def weighted_min_conflict_fitness(self, individual):
        score = 0
        col_weight = 1
        cage_weight = 2

        for i in range(self.size):
            score += col_weight * (len(set(individual[j][i] for j in range(self.size))) / self.size)

        for cage, (operation, target) in zip(self.cages, self.operations):
            values = [individual[x][y] for x, y in cage]
            if operation == 'add':
                score += cage_weight * (1 - abs(sum(values) - target) / target)
            elif operation == 'multiply':
                score += cage_weight * (1 - abs(np.prod(values) - target) / target)
            elif operation == 'subtract' and len(values) == 2:
                score += cage_weight * (1 - abs(abs(values[0] - values[1]) - target) / target)
                if abs(values[0] - values[1]) == target:
                    score += cage_weight
            elif operation == 'divide' and len(values) == 2:
                max_val, min_val = max(values), min(values)
                if min_val != 0:
                    score += cage_weight * (1 - abs(abs(values[0] - values[1]) - target) / target)
                    if (max_val/min_val) == target:
                        score += cage_weight
            elif operation == 'none':
                if values[0] == target:
                    score += cage_weight
        return score

    def is_goal_state(self, individual):
        score = 0
        for i in range(self.size):
            if len(set(individual[i])) == self.size:
                score += 1
            if len(set(individual[j][i] for j in range(self.size))) == self.size:
                score += 1

        for cage, (operation, target) in zip(self.cages, self.operations):
            values = [individual[x][y] for x, y in cage]
            if operation == 'add' and sum(values) == target:
                score += 1
            elif operation == 'multiply' and np.prod(values) == target:
                score += 1
            elif operation == 'subtract' and len(values) == 2 and abs(values[0] - values[1]) == target:
                score += 1
            elif operation == 'divide' and len(values) == 2 and (max(values) / min(values) == target):
                score += 1
            elif operation == 'none' and values[0] == target:
                score += 1

        return score == self.size * 2 + len(self.cages)

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.size):
            if random.random() < 0.5:
                child.append(parent1[i][:])
            else:
                child.append(parent2[i][:])
        return child

    def mutate(self, individual):
        for i in range(self.size):
            if random.random() < self.mutation_rate:
                j, k = random.sample(range(self.size), 2)
                individual[i][j], individual[i][k] = individual[i][k], individual[i][j]

        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.size), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def solve(self, timeout=None):
        best_fitness = 0
        stagnation_counter = 0

        for generation in range(self.generations):
            self.population.sort(
                key=lambda x: self.weighted_min_conflict_fitness(x),
                reverse=True
            )
            current_best_fitness = self.weighted_min_conflict_fitness(self.population[0])

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if self.is_goal_state(self.population[0]):
                return True, self.population[0], generation

            if stagnation_counter > 100:
                return False, self.population[0], generation

            new_population = self.population[:2]

            while len(new_population) < self.population_size:
                parent1, parent2 = random.choices(self.population[:20], k=2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

        return False, self.population[0], self.generations
