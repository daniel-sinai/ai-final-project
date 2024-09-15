import time


class DFSSolver:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, timeout=None):
        start_time = time.time()
        fringe = []
        visited = set()

        start_state = self.problem.get_initial_state()
        fringe.append(start_state)

        while fringe:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError

            current_state = fringe.pop()

            if self.problem.is_goal_state(current_state):
                return current_state

            if current_state in visited:
                continue

            visited.add(current_state)

            for successor in self.problem.get_successors(current_state):
                if successor not in visited:
                    fringe.append(successor)
        return None
