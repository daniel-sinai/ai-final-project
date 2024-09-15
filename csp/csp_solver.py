import heapq
import random
import time
import numpy as np
from csp.csp import CalcudokuCSP


class BacktrackingCSPSolver:
    def __init__(self, board_manager, select_unassigned_var_type="none",
                 order_domain_values_type="random", use_fc=False, use_ac=False):
        self.use_arc_consistency = use_ac
        self.use_forward_checking = use_fc
        self.select_unassigned_var_type = select_unassigned_var_type
        self.order_domain_values_type = order_domain_values_type
        self.backtracks_measurement = 0
        self.assignments_measurement = 0
        self.conflict_checks_measurement = [0]
        self.csp = CalcudokuCSP(
            board_manager.size_,
            board_manager.cages_,
            board_manager.operations_,
            self.conflict_checks_measurement
        )

    def solve(self, timeout_seconds=None):
        start_time = time.time()
        index = 0
        domains = [{var: var.starter_domain.copy() for var in self.csp.X}]
        unassigned = [v for v in self.csp.X if v.get_assignment() == 0]
        prev_vars = []
        var = None
        while True:
            if timeout_seconds is not None and (time.time() - start_time) > timeout_seconds:
                raise TimeoutError
            cur_domains = {var: domains[-1][var].copy() if var in unassigned else [var.assignment] for var in self.csp.X}
            arc_valid, fc_valid, consistent_value, updated_domains = True, True, None, None
            if not self.use_arc_consistency or self._arc_consistency(cur_domains):
                if var is None:
                    var = self._select_unassigned_var(unassigned, cur_domains)
                for consistent_value in self._order_domain_values(var, cur_domains, unassigned):
                    if not self.use_forward_checking:
                        updated_domains = cur_domains
                        break
                    fc_valid, updated_domains = self._forward_checking(var, consistent_value,
                                                                       {var: cur_domains[var].copy() for var in
                                                                        cur_domains.keys()})
                    if fc_valid:
                        break
                    else:
                        consistent_value = None

            if consistent_value:  # continue to next
                domains[-1][var] = cur_domains[var].copy()
                var.assignment = consistent_value
                unassigned.remove(var)
                self.assignments_measurement += 1
                if not unassigned:  # finished - solution found
                    return self._get_board(), self.backtracks_measurement, self.assignments_measurement, self.conflict_checks_measurement[0]
                prev_vars.append(var)
                var = None
                domains.append(updated_domains)
            else:  # backtrack
                if not prev_vars:
                    return None, self.backtracks_measurement, self.assignments_measurement, self.conflict_checks_measurement[0]
                domains.pop()
                var = prev_vars.pop()
                # We need to delete the current assignment of prev cell var (the one that we came from in backtracking)
                # so we will know not to use it again
                var.reset_assignment()  # var.assignment = 0
                unassigned.append(var)
                self.backtracks_measurement += 1
            index += 1

    def _order_domain_values(self, var, domains, unassigned):
        if self.order_domain_values_type == "random":
            return self._select_value(var, domains, unassigned)
        if self.order_domain_values_type == "lcv":
            return self._select_value_LCV(var, domains, unassigned)

    def _select_value_LCV(self, var, domains, unassigned):
        # First, we find values that are not in conflict with the current assignment
        valid_values = []
        domain = domains[var]
        for d in domain:
            if not self.is_assignment_conflict_with_prev_vars(var, d, domain):
                valid_values.append(d)
        domains[var] = valid_values

        # Then, we take the value that will potentially cause the least conflicts with future assignments
        value_to_constraints_heap = []
        for val in valid_values:
            valid, conflicts_amount = self._remaining_vars_conflicts_amount(var, domains, val)
            if valid:
                heapq.heappush(value_to_constraints_heap, (conflicts_amount, val))

        while value_to_constraints_heap:
            val = heapq.heappop(value_to_constraints_heap)[1]
            domains[var].remove(val)
            yield val

    def _remaining_vars_conflicts_amount(self, var, domains, assignment):
        old_assignment = var.assignment
        var.assignment = assignment
        counter = 0
        for constraint in self.csp.C:
            var1, var2 = constraint.node1, constraint.node2
            if var1 == var and var2.get_assignment() == 0:
                var2_domain = domains[var2]
                var2_domain_len = len(var2_domain)
                for dj in var2_domain:
                    var2.assignment = dj
                    self.conflict_checks_measurement[0] += 1
                    if constraint.has_conflict():
                        counter += 1
                        var2_domain_len -= 1
                    var2.reset_assignment()  # var2.assignment = 0
                if var2_domain_len == 0:
                    var.assignment = old_assignment
                    var2.reset_assignment()
                    return False, None
        var.assignment = old_assignment
        return True, counter

    def is_assignment_conflict_with_prev_vars(self, var, assignment, domain):
        old_assignment = var.assignment
        var.assignment = assignment
        for c in self.csp.C:
            is_assigned = c.are_nodes_assigned()
            if is_assigned:
                self.conflict_checks_measurement[0] += 1
                if c.has_conflict():
                    var.assignment = old_assignment
                    return True

        var.assignment = old_assignment
        return False

    def _select_value(self, var, domains, unassigned):
        domain = domains[var]
        while domain:
            value = domain.pop(random.randint(0, len(domain) - 1))
            if not self.is_assignment_conflict_with_prev_vars(var, value, domain):
                yield value

    def _select_unassigned_var(self, unassigned, domains):
        if self.select_unassigned_var_type == "none":
            return self._select_unassigned_var_none(unassigned, domains)
        if self.select_unassigned_var_type == "mrv":
            return self._select_unassigned_var_mrv(unassigned, domains)

    def _select_unassigned_var_none(self, unassigned, domains):
        return unassigned[0]

    def _select_unassigned_var_mrv(self, unassigned, domains):
        some_min_domain_var = min(unassigned, key=lambda var: len(domains[var]))
        min_domain_length = len(domains[some_min_domain_var])
        min_vars = [var for var in unassigned if len(domains[var]) == min_domain_length]
        if len(min_vars) == 1:
            return min_vars[0]

        # degree tie-breaker
        num_of_constraints_per_var = {var: 0 for var in min_vars}
        for constraint in self.csp.C:
            var1, var2 = constraint.node1, constraint.node2
            if var1 in min_vars and var2 in unassigned:
                num_of_constraints_per_var[var1] += 1
            if var2 in min_vars and var1 in unassigned:
                num_of_constraints_per_var[var2] += 1

        return max(num_of_constraints_per_var, key=num_of_constraints_per_var.get)

    def _forward_checking(self, var, assignment, domains_copy):
        old_assignment = var.assignment
        var.assignment = assignment
        for c in self.csp.C:
            if c.node1 != var or c.node2.get_assignment() != 0:
                continue
            var2 = c.node2
            var2_domain = domains_copy[var2]
            for d in var2_domain[:]:
                var2.assignment = d
                is_assigned = c.are_nodes_assigned()
                if is_assigned:
                    self.conflict_checks_measurement[0] += 1
                    if c.has_conflict():
                        var2_domain.remove(d)
            var2.reset_assignment()
            if not domains_copy[var2]:
                var.assignment = old_assignment
                return False, None

        var.assignment = old_assignment
        return True, domains_copy

    def _get_board(self):
        board = np.zeros((self.csp.size, self.csp.size), dtype=int)
        for loc, var in self.csp.cell_vars.items():
            board[loc[0], loc[1]] = var.get_assignment()
        return board

    def _arc_consistency(self, cur_domains):
        queue = [c for c in self.csp.C]
        while queue:
            constraint = queue.pop(0)
            var1, var2 = constraint.node1, constraint.node2
            if self._revise(constraint, cur_domains):
                if not cur_domains[var1]:
                    return False
                for constraint1 in self.csp.C:
                    var3, var4 = constraint1.node1, constraint1.node2
                    if var4 == var1 and var3 != var2:
                        if constraint1 not in queue:
                            queue.append(constraint1)
        return True

    def _revise(self, constraint, domains_copy):
        var1, var2 = constraint.node1, constraint.node2
        revised = False
        old_var1_assignment = var1.assignment
        old_var2_assignment = var2.assignment

        for d1 in domains_copy[var1][:]:
            var1.assignment = d1
            does_have_consistent_assignment = False
            for d2 in domains_copy[var2]:
                var2.assignment = d2
                self.conflict_checks_measurement[0] += 1
                if not constraint.has_conflict():
                    does_have_consistent_assignment = True
                    break
            if not does_have_consistent_assignment:
                domains_copy[var1].remove(d1)
                revised = True

        var1.assignment = old_var1_assignment
        var2.assignment = old_var2_assignment
        return revised
