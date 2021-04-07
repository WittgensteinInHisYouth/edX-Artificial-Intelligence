import sys
import numpy as np
import pandas as pd

class AC3(object):
    def __init__(self, sudoku_str):
        X, D = self.CSPFormulation(sudoku_str)
        self.vars = X
        self.domains =D
        self.assignment = pd.DataFrame(index=range(9), columns=range(9))

    def CSPFormulation(self, sudoku_str):

        sudoku_board = np.array(list(sudoku_str), dtype=int).reshape((9, 9))
        vars = [(i, j) for i in range(9) for j in range(9)]

        domains = {var: list(range(1, 10)) for var in vars}

        for var in vars:
            if sudoku_board[var] != 0:
                domains[var] = [sudoku_board[var]]

        # self.display(vars, domains)
        return vars, domains


    def constrainQ(self, locationA, locationB):

        # Given two locations, such as (0,0) and (9,9), determine whether or not they are constrained.

        (x1, y1) = locationA
        (x2, y2) = locationB

        row_blk_A = x1 // 3
        col_blk_A = y1 // 3
        row_blk_B = x2 // 3
        col_blk_B = y2 // 3

        return (x1 == x2) or (y1 == y2) or (row_blk_A == row_blk_B and col_blk_A == col_blk_B)

    def neighbors(self, var):

        neighbors_ls = []

        for candidate in self.vars:
            if self.constrainQ(var, candidate) and candidate != var:
                neighbors_ls.append(candidate)

        return neighbors_ls


    def revise(self, x_i, x_j):
        revised = False
        assert  x_i != x_j
        D_i, D_j = self.domains[x_i], self.domains[x_j]

        for x in D_i:

            noValueAdmissbleQ = True
            for y in D_j:
                if x != y:
                    noValueAdmissbleQ = False
                    break

            if noValueAdmissbleQ:
                self.domains[x_i].remove(x)
                # print(f'Trigger. Remove {x} from domain of {x_i}')
                revised = True

        return revised

    def display(self, vars, domains):
        df = pd.DataFrame(index=range(9), columns=range(9))
        for var in vars:
            df.loc[var] = domains[var]

        return df

    def run(self):

        queue = []

        for var1 in self.vars:
            for var2 in self.vars:
                if var1 != var2 and self.constrainQ(var1, var2) and {var1, var2} not in queue:
                    queue.append((var1, var2))

        while queue:
            (X_i, X_j) = queue.pop(0)
            if self.revise(X_i, X_j):
                if not self.domains[X_i]:
                    return False

                for X_k in self.neighbors(X_i):
                    if X_k != X_j:
                        queue.append((X_k, X_i))

        self.assignment = self.display(self.vars, self.domains)

        return True


class BTS(object):
    def __init__(self, sudoku_str):

        X, D = self.CSPFormulation(sudoku_str)
        self.vars = X
        self.domains = D
        self.assignment = pd.DataFrame(index=range(9), columns=range(9))

    def CSPFormulation(self, sudoku_str):

        sudoku_board = np.array(list(sudoku_str), dtype=int).reshape((9, 9))
        vars = [(i, j) for i in range(9) for j in range(9)]
        domains = {var: list(range(1, 10)) for var in vars}

        for var in vars:
            if sudoku_board[var] != 0:
                domains[var] = [sudoku_board[var]]

        return vars, domains

    def constrainQ(self, locationA, locationB):

        # Given two locations, such as (0,0) and (9,9), determine whether or not they are constrained.

        (x1, y1) = locationA
        (x2, y2) = locationB

        row_blk_A = x1 // 3
        col_blk_A = y1 // 3
        row_blk_B = x2 // 3
        col_blk_B = y2 // 3

        return (x1 == x2) or (y1 == y2) or (row_blk_A == row_blk_B and col_blk_A == col_blk_B)

    def completeQ(self, assignment_df):
        print('#################')
        print(assignment_df)
        print('#################')

        return not assignment_df.isnull().values.any()


    def getRemainingValues(self, assignment):

        updatedDomain = self.domains.copy()
        # print('#################')
        # print('inital domain',  self.domains.copy())
        # print(assignment)
        for var in self.vars:

            value = assignment.loc[var]

            if not np.isnan(value):

                del updatedDomain[var]

                for other_var in self.vars:
                    if other_var != var and (other_var in updatedDomain) and self.constrainQ(var, other_var):


                        if value in updatedDomain[other_var]:

                            new_val = updatedDomain[other_var][:]
                            new_val.remove(value)
                            updatedDomain[other_var] = new_val

        # Forward Checking
        for key, val in updatedDomain.copy().items():
            if val == []:
                return None

        return updatedDomain




    def selectUnassignedVariable(self, assignment):
        remaining_dict = self.getRemainingValues(assignment)
        if remaining_dict is None:
            return None, None
        shortest_remaining_length = np.inf

        for remaining_var, remaining_value in remaining_dict.items():
            if 0 < len(remaining_value) < shortest_remaining_length:
                shortest_candiate = remaining_var
                shortest_remaining_length = len(remaining_value)
        try:
            return shortest_candiate, remaining_dict[shortest_candiate]
        except UnboundLocalError:
            return None, None


    def consistentQ(self, assignment_df):

        for var1 in self.vars:
            for var2 in self.vars:
                if var1 != var2 and self.constrainQ(var1, var2):
                    val1, val2 = assignment_df.loc[var1], assignment_df.loc[var2]
                    if (not np.isnan(val1)) and (not np.isnan(val2)) and (val1 == val2):
                        print('inconsistnet')
                        print(assignment_df)
                        return False
        return True

    def backtrack(self):
        if self.completeQ(self.assignment):
            return self.assignment


        var, values = self.selectUnassignedVariable(self.assignment)
        if var is None:
            return None

        for value in  values:
            temp_assignment = self.assignment.copy(deep=True)
            temp_assignment.loc[var] = value
            if self.consistentQ(temp_assignment):
                self.assignment.loc[var] = value
                result = self.backtrack()
                if result is not None:
                    return result
            self.assignment.loc[var] = np.nan

        return None

    def backtrack_search(self):
        return self.backtrack()





def solved(assignment_df):
    if np.all(assignment_df.applymap(lambda x: len(x) == 1).values):
        
        solution = []
        num_row, num_col = assignment_df.shape
        
        for row_ix in range(num_row):
            for col_ix in range(num_col):
                solution.append(str(assignment_df.loc[row_ix, col_ix][0]))

        if '0' in solution:
            raise Exception('Unsolved entry exists.')
        elif len(solution) != 81:
            raise Exception('Missing entry')

        return ''.join(solution)

    else:
        return False

def main():

    sudoku_string = sys.argv[1].lower()

    ac3 = AC3(sudoku_string)
    successQ = ac3.run()

    if successQ:

        solution = solved(ac3.assignment)
        if solution:
            f = open('output.txt', 'w')
            f.write(solution + ' AC3')
            f.close()

            return None

    assignment = BTS(sudoku_string).backtrack_search()
    solution = []
    num_row, num_col = assignment.shape

    for row_ix in range(num_row):
        for col_ix in range(num_col):
            solution.append(str(assignment.loc[row_ix, col_ix]))

    if '0' in solution:
        raise Exception('Unsolved entry exists.')
    elif len(solution) != 81:
        raise Exception('Missing entry')

    solution = ''.join(solution)

    f = open('output.txt', 'w')
    f.write(solution + ' BTS')
    f.close()

    return None

if __name__ == '__main__':

    main()