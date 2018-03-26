from sudoku import Sudoku

class CSP(object):
    '''
    This is a helper class to organize the problem that needs to be solved 
    '''
    def __init__(self, variables, domains):
        '''
        
        :param variables: list of variables that need to be assigned 
        :param domains: the possible values for each variable [(var, vals)]
        '''
        self.variables = variables
        self.domains = domains
        self.constraints = self.init_constraints()

    def init_constraints(self):
        '''
        Your choice on how to implement constraints. 
        HINT: You should define constraints such they are fast to evaluate. One method is to define the constraints as a 
                function of the row and column. Sets are a fast way to remove duplicates.
        :return: your choice
        '''
        constraints = []
        return constraints


class CSP_Solver(object):
    """
    This class is used to solve the CSP with backtracking using the minimum value remaining heuristic.
    HINT: you will likely want to implement functions in the backtracking sudo code in figure 6.5 in the text book.
            We have provided some prototypes that might be helpful. You are not required to use any functions defined
            here and can modify any function other than the solve method. We will test your code with the solve method
            and so it must have no parameters and return the type it says. 
         
    """
    def __init__(self, puzzle_file):
        '''
        Initialize the solver instance. The lower the number of the puzzle file the easier it is. 
        It is a good idea to start with the easy puzzles and verify that your solution is correct manually. 
        You should run on the hard puzzles to make sure you aren't violating corner cases that come up.
        Harder puzzles will take longer to solve.
        :param puzzle_file: the puzzle file to solve 
        '''
        self.sudoku = Sudoku(puzzle_file) # this line has to be here to initialize the puzzle

        # you can use these if you want
        self.num_guesses = 0
        vars = []
        domains = []
        self.csp = CSP(vars, domains)

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver = CSP_Solver('puz-001.txt')
    ### solved_board, num_guesses = csp_solver.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):
        '''
        This method solves the puzzle initialized in self.sudoku 
        You should define backtracking search methods that this function calls
        The return from this function NEEDS to match the correct type
        Return None, number of guesses no solution is found
        :return: tuple (list of list (ie [[]]), number of guesses
        '''

        return [[]], 0

    def backtracking_search(self, sudoku, csp):
        '''
        This function might be helpful to initialize a recursive backtracking search function
        You do not have to use it.
        
        :param sudoku: Sudoku class instance
        :param csp: CSP class instance
        :return: board state (list of lists), num guesses 
        '''
        return self.recursive_backtracking(sudoku, csp), self.num_guesses

    def recursive_backtracking(self, sudoku, csp):
        '''
        recursive backtracking search function.
        You do not have to use this
        :param sudoku: Sudoku class instance
        :param csp: CSP class instance
        :return: board state (list of lists)
        '''
        return None

    def select_unassigned_var(self, board):
        '''
        Function that should select an unassigned variable to assign next
        You do not have to use this
        :param board: list of lists
        :return: variable
        '''
        return None # replace with your choice

    def order_domain_values(self, var, csp):
        '''
        A function to return domain values for a variable.
        You do not need to use this. 
        :param var: variable
        :param csp: CSP problem instance
        :return: list of domain values for var
        '''
        return None

    def consistent(self, var, value, board, constraints):
        '''
        This function checks to see if assigning value to var on board violates any of the constraints
        You do not need to use this function
        :param var: variable to be assigned, tuple (row col) 
        :param value: value to assign to var
        :param board: board state (list of list)
        :param constraints: to check to see if they are violated
        :return: True if consistent False otherwise
        '''
        return False

if __name__ == '__main__':
    csp_solver = CSP_Solver('puz-001.txt')
    solution, guesses = csp_solver.solve()

