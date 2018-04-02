from sudoku import Sudoku
import numpy as np
import math

class CSP(object):
    '''
    This is a helper class to organize the problem that needs to be solved
    '''
    def __init__(self, variables, domains):
        '''

        :param variables: list of variables that need to be assigned
        :param domains: the possible values for each variable [(var, vals)]
        '''
        domains = []
        for i in range(0,9*9):
            domains.append([1,2,3,4,5,6,7,8,9])

        self.variables = variables
        self.domains = domains                              #81 of range(1,10), representing total choice for all cells


    def constraints(self):
        '''
        Your choice on how to implement constraints.
        HINT: You should define constraints such they are fast to evaluate. One method is to define the constraints as a
                function of the row and column. Sets are a fast way to remove duplicates.
        :return: False if one spec is not met, true if all specs met, runtime is low when board is not complete
        '''
        tester = [1,2,3,4,5,6,7,8,9]                        #list of all needed num
        temp_list =[]
        for j in range(0,9):
            for i in self.variables[j]:
                if i == 0:
                    return False                                #simple check, is all filled?
        """
        if all filled then do the ultimate test
        row check
        col check
        box check
        """
        for a in range(0,9):                                #row check
            if np.unique(self.variables[a]).size != len(tester):
                return False
            if np.unique(self.variables, axis = a).size != self.variables.size:
                return False

        for R in range(0,3):                                #all 9 sub blocks
            for C in range(0,3):


                for r in range(0,3):                        #record all appearing num
                    for c in range(0,3):
                        if self.variables[r][c] not in temp_list:
                            temp_list.append(self.variables[r][c])
                        else:
                            return False
                if temp_list.size != len(tester):
                    return False
                else:
                    temp_list.clear()

        return True


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
        vars = np.zeros(shape = [9,9], dtype = int)
        #vars = self.sudoku.load_board(puzzle_file)      #read board config into vars
        # you can use these if you want
        self.num_guesses = 0

        for i in range(0,9):
            vars[i] = self.sudoku.load_board(puzzle_file)[i]      #read board config into vars
            print(vars[i])                                             #print the original config
        domains = [1,2,3,4,5,6,7,8,9]
        print(vars)
        for b in range(0,3):
            print(b)
        print(domains)
        self.csp = CSP(vars, domains)                                  #the CSP

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver = CSP_Solver('puz-001.txt')
    ### solved_board, num_guesses = csp_solver.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):        ##
        '''
        This method solves the puzzle initialized in self.sudoku
        You should define backtracking search methods that this function calls
        The return from this function NEEDS to match the correct type
        Return None, number of guesses no solution is found
        :return: tuple (list of list (ie [[]]), number of guesses
        '''

        return self.csp.variables, self.csp.domains

    def backtracking_search(self, sudoku, csp):
        '''
        This function might be helpful to initialize a recursive backtracking search function
        You do not have to use it.

        :param sudoku: Sudoku class instance
        :param csp: CSP class instance
        :return: board state (list of lists), num guesses
        '''
        return self.recursive_backtracking(sudoku, csp), self.num_guesses+1

    def recursive_backtracking(self, sudoku, csp):
        '''
        recursive backtracking search function.
        You do not have to use this
        :param sudoku: Sudoku class instance
        :param csp: CSP class instance
        :return: search result T or F
        '''
        value = 0
        if csp.constraints():                       #iff complete and OK then quit
            return csp

        row, col = self.select_unassigned_var(csp.variables)        #next empty cell
        if row == 10 or col == 10:
            if csp.constraints():                       #iff complete and OK then quit
                return csp, True
            else:
                return csp, False                        #board not satisfying constraints, next try
        else:
            while self.consistent(row, col, value, csp.variables):      #find a consistent value for this cell
                value = self.Order_Domain_Values(col,row,csp)
            csp.variables[row][col] = value             #update

            recursive_backtracking(self. sudoku, csp)



        return csp, False

    def Order_Domain_Values(self , col, row, csp):
        cell = col + row*9                              #which cell we are talking about
        value = csp.domains[cell][0]                    #pick a num from the possible choice and remove it
        csp.domains[cell],remove(value)


        return value



    def select_unassigned_var(self, board):
        '''
        find the next empty cell
        :param board: list of lists
        :return: coordinates of the empty cell, or 10, 10 for NOT FOUND
        '''
        #scan through the list and return first cell with 0
        found = True
        for i in range(0,9):            #row
            for j in range(0,9):        #col
                if board[i][j] == 0:
                    found = False
                    return [i,j]       #return
        if not found:
            return [10,10]


    def consistent(self, row,col , value, board):
        '''
        This function checks to see if assigning value to var on board violates any of the constraints
        You do not need to use this function
        :param var: variable to be assigned, tuple (row col)
        :param value: value to assign to var
        :param board: board state (list of list)
        :param constraints: to check to see if they are violated
        :return: True if consistent False otherwise
        '''
        cons = True                               #default True

        block_r = row // 3                        #calculate start r of block
        """
        Three checks:
        1.row check
        2.col check
        3.3*3 check
        """
        if value == 0 :
            cons == False
            break
        else:
            for a in range(0,9):                                    # row check
                if value == board[row][a] :
                    cons = False
                    break

            for b in range(0,9):                                    # col check
                if value == board[b][col]:
                    cons = False
                    break

            for r in range(0,3):                                    # 3*3 check
                for c in range(0,3):
                    if value == board[block_r*3+r][block_c*3+c]:
                        cons = False
                        break

        return cons





if __name__ == '__main__':
    csp_solver = CSP_Solver('puz-001.txt')
    solution, guesses = csp_solver.solve()
