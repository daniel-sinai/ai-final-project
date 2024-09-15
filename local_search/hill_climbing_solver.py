class HillClimbingSolver:
    def __init__(self, problem, iterations=50):
        self.iterations = iterations
        self.problem = problem

    def solve(self, callback=None):
        current = self.problem.get_initial_state()

        for i in range(self.iterations):
            if callback:
                callback(current)
            neighbor = self.problem.get_lowest_valued_successor(current)
            if neighbor.value >= current.value:
                break
            current = neighbor
        return current
