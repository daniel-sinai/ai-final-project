import itertools


OP_TO_LAMBDA = {"add": lambda x, y: x + y,
                "subtract": lambda x, y: abs(x - y),
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: max(x, y) // min(x, y),
                "none": lambda x, y: x}


class Variable:
    def __init__(self, csp, first_var=None, second_var=None, starter_domain=None, x_ind=None):
        self.first_var = first_var
        self.second_var = second_var
        self.starter_domain = starter_domain
        self.assignment = 0
        self.csp = csp
        self.x_ind = x_ind

    def __str__(self):
        if self.x_ind is not None:
            return f"x{self.x_ind}=" + str(self.assignment)
        else:
            return f"[{self.first_var} | {self.second_var}]=" + str(self.assignment)

    def get_assignment(self):
        if isinstance(self.assignment, tuple):
            return self.assignment[2]
        else:
            return self.assignment

    def reset_assignment(self):
        if isinstance(self.assignment, tuple):
            self.assignment = (0, 0, 0)
        else:
            self.assignment = 0


class Constraint:
    def __init__(self, csp, node1, node2, constraint_checking, are_nodes_assigned_func):
        self.node1 = node1  # Y1 (<x1,x2>)
        self.node2 = node2  # Y1 (<x1,x2>)
        self.constraint_checking = constraint_checking  # lambda var1, var2: var1.cells[0].assignment + var1.cells[1].assignment = var1.assignment
        self.are_nodes_assigned_func = are_nodes_assigned_func
        self.csp = csp

    def has_conflict(self):
        return not self.constraint_checking(self.node1, self.node2)

    def are_nodes_assigned(self):
        return self.are_nodes_assigned_func(self.node1, self.node2)

    def __str__(self):
        return f"Constraint: ({self.node1.__str__()}-{self.node2.__str__()})"


class CalcudokuCSP:
    def __init__(self, size, cages, operations, conflict_checks_measurement):
        self.size = size
        self.X = []
        self.cell_vars = dict()
        self.C = list()
        self.cages = cages
        self.operations = operations
        self.conflict_checks_measurement = conflict_checks_measurement

        self.initialize_cells_vars()
        self.initialize_row_col_constraints()
        self.initialize_cages_vars_and_constraints()

    def initialize_cells_vars(self):
        for i in range(self.size):
            for j in range(self.size):
                var = Variable(self, starter_domain=[i for i in range(1, self.size + 1)], x_ind=i * self.size + j)
                self.X.append(var)
                self.cell_vars[(i, j)] = var

    def initialize_row_col_constraints(self):
        for row in range(self.size):
            for col in range(self.size):
                var = self.cell_vars[(row, col)]
                for k in range(self.size):
                    if row != k:
                        self.C.append(Constraint(
                            self, var, self.cell_vars[(k, col)],
                            lambda node1, node2: node1.assignment != node2.assignment,
                            lambda node1, node2: node1.assignment != 0 and node2.assignment != 0))
                    if col != k:
                        self.C.append(Constraint(
                            self, var, self.cell_vars[(row, k)],
                            lambda node1, node2: node1.assignment != node2.assignment,
                            lambda node1, node2: node1.assignment != 0 and node2.assignment != 0))

    def initialize_cages_vars_and_constraints(self):
        for cage, operation in zip(self.cages, self.operations):
            base_vars = [self.cell_vars[loc] for loc in cage]
            op, target = operation
            if len(base_vars) == 1:
                base_vars[0].starter_domain = [target]
                base_vars[0].assignment = target
                self.C.append(Constraint(
                    self,
                    base_vars[0],
                    base_vars[0],
                    lambda node1, _, target1=target: node1.assignment == target1,
                    lambda node1, _: node1.assignment != 0))
            else:
                self.add_cage_aux_vars(base_vars, op, target, 1, True)

    def get_var_locs(self, var):
        vars = []

        def helper(var):
            if var.first_var is None:
                for loc, var2 in self.cell_vars.items():
                    if var == var2:
                        vars.append(f"X{loc[0] * self.size + loc[1]}")
                        return

            helper(var.first_var)
            helper(var.second_var)

        helper(var)

        return vars

    def init_aux_domain(self, target, op, underlying_vars_amount, aux_vars_amount_in_level):
        min_val, max_val = None, None
        if op == 'multiply':
            max_mult = self.size ** underlying_vars_amount
            max_val = min(max_mult, target)
            min_val = 1
        elif op == 'add':
            max_add = self.size * underlying_vars_amount
            max_val = min(max_add, target)
            min_val = underlying_vars_amount
        else:
            raise 'init aux domain illegal operation [only * +]'
        aux_domain = []
        for (value1, value2) in itertools.product(range(1, target+1), repeat=2):
            result = OP_TO_LAMBDA[op](value1, value2)
            if min_val <= result <= max_val:
                aux_domain.append((value1, value2, result))
        return aux_domain

    def add_cage_aux_vars(self, vars, op, target, aux_vars_amount_in_level, finish):
        if len(vars) == 1:
            return vars[0]

        mid = len(vars) // 2
        first_half = vars[:mid]
        second_half = vars[mid:]

        var1 = self.add_cage_aux_vars(first_half, op, target, aux_vars_amount_in_level*2, False)
        var2 = self.add_cage_aux_vars(second_half, op, target, aux_vars_amount_in_level*2, False)

        if finish:
            self.C.append(Constraint(
                self,
                var1,
                var2,
                lambda node1, node2: OP_TO_LAMBDA[op](node1.get_assignment(), node2.get_assignment()) == target,
                lambda node1, node2: node1.get_assignment() != 0 and node2.get_assignment() != 0))
            self.C.append(Constraint(
                self,
                var2,
                var1,
                lambda node1, node2: OP_TO_LAMBDA[op](node1.get_assignment(), node2.get_assignment()) == target,
                lambda node1, node2: node1.get_assignment() != 0 and node2.get_assignment() != 0))
            return

        aux_domain = self.init_aux_domain(target, op, len(vars), aux_vars_amount_in_level)
        aux_var = Variable(self, var1, var2, aux_domain)
        aux_var.assignment = (0, 0, 0)
        self.X.append(aux_var)
        # ****** Create Constraints ******
        # ****** Constraint : var1 -> aux_var ******
        self.C.append(Constraint(
            self,
            var1,
            aux_var,
            lambda v, aux_v: aux_v.assignment[0] == v.get_assignment(),
            lambda v, aux_v: v.get_assignment() != 0 and aux_v.assignment[0] != 0))
        # ****** Constraint : aux_var -> var ******
        self.C.append(Constraint(
            self,
            aux_var,
            var1,
            lambda aux_v, v: aux_v.assignment[0] == v.get_assignment(),
            lambda aux_v, v: v.get_assignment() != 0 and aux_v.assignment[0] != 0))
        # ****** Constraint : var2 -> aux_var ******
        self.C.append(Constraint(
            self,
            var2,
            aux_var,
            lambda v, aux_v: aux_v.assignment[1] == v.get_assignment(),
            lambda v, aux_v: v.get_assignment() != 0 and aux_v.assignment[1] != 0))
        # ****** Constraint : aux_var -> var2 ******
        self.C.append(Constraint(
            self,
            aux_var,
            var2,
            lambda aux_v, v: aux_v.assignment[1] == v.get_assignment(),
            lambda aux_v, v: v.get_assignment() != 0 and aux_v.assignment[1] != 0))
        # ****** Constraint : aux_var -> aux_var ******
        self.C.append(Constraint(
            self,
            aux_var,
            aux_var,
            lambda aux_v, _: OP_TO_LAMBDA[op](aux_v.assignment[0],
                                              aux_v.assignment[1]) == aux_v.assignment[2],
            lambda aux_v, _: aux_v.assignment[0] != 0 and aux_v.assignment[1] != 0 and aux_v.assignment[2] != 0))
        return aux_var
