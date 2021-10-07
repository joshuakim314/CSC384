#Look for #IMPLEMENT tags in this file.
'''
All models need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = futoshiki_csp_model_1(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the Futoshiki puzzle.

1. futoshiki_csp_model_1 (worth 20/100 marks)
    - A model of a Futoshiki grid built using only 
      binary not-equal constraints for both the row and column constraints.

2. futoshiki_csp_model_2 (worth 20/100 marks)
    - A model of a Futoshiki grid built using only n-ary 
      all-different constraints for both the row and column constraints. 

'''
from cspbase import *
import itertools

def futoshiki_csp_model_1(futo_grid):
    ##IMPLEMENT
    board = generate_variables(futo_grid)
    constraints = []
    constraints += generate_inequality_constraints(futo_grid, board, True)
    constraints += generate_row_constraints(board, True)
    constraints += generate_col_constraints(board, True)
    csp = CSP("Futoshiki Model 1", vars=list(itertools.chain(*board)))
    for c in constraints:
        csp.add_constraint(c)
    return csp, board
    

def futoshiki_csp_model_2(futo_grid):
    ##IMPLEMENT
    board = generate_variables(futo_grid)
    constraints = []
    constraints += generate_inequality_constraints(futo_grid, board, False)
    constraints += generate_row_constraints(board, False)
    constraints += generate_col_constraints(board, False)
    csp = CSP("Futoshiki Model 2", vars=list(itertools.chain(*board)))
    for c in constraints:
        csp.add_constraint(c)
    return csp, board
    

def generate_variables(futo_grid):
    board = []
    n = len(futo_grid)
    for r, row in enumerate(futo_grid):
        vars = []
        for c in range(0, 2*n, 2):
            dom = [row[c]] if row[c] else list(range(1, n+1))
            vars.append(Variable('V{}{}'.format(r, int(c/2)), dom))
        board.append(vars)
    return board


def generate_inequality_constraints(futo_grid, board, is_binary):
    constraints = []
    n = len(board)
    for i in range(n):
        for j in range(n-1):
            inequality = futo_grid[i][2*j+1]
            if inequality != '.' or is_binary:
                x = board[i][j]
                y = board[i][j+1]
                c = Constraint('C({}, {})'.format(x.name, y.name), [x, y])
                c.add_satisfying_tuples(generate_binary_sat_tuples(x.cur_domain(), y.cur_domain(), inequality))
                constraints.append(c)
    return constraints


def generate_row_constraints(board, is_binary):
    constraints = []
    n = len(board)
    if is_binary:
        for i in range(n):
            for j in range(n-2):
                for k in range(j+2, n):
                    x = board[i][j]
                    y = board[i][k]
                    c = Constraint('C({}, {})'.format(x.name, y.name), [x, y])
                    c.add_satisfying_tuples(generate_binary_sat_tuples(x.cur_domain(), y.cur_domain()))
                    constraints.append(c)
    else:
        for i in range(n):
            c = Constraint('C(Row {})'.format(i), [x for x in board[i]])
            c.add_satisfying_tuples(generate_all_diff_sat_tuples(*[x.cur_domain() for x in board[i]]))
            constraints.append(c)
    return constraints


def generate_col_constraints(board, is_binary):
    constraints = []
    n = len(board)
    if is_binary:
        for i in range(n-1):
            for j in range(n):
                for l in range(i+1, n):
                    x = board[i][j]
                    y = board[l][j]
                    c = Constraint('C({}, {})'.format(x.name, y.name), [x, y])
                    c.add_satisfying_tuples(generate_binary_sat_tuples(x.cur_domain(), y.cur_domain()))
                    constraints.append(c)
    else:
        for j in range(n):
            c = Constraint('C(Col {})'.format(j), [row[j] for row in board])
            c.add_satisfying_tuples(generate_all_diff_sat_tuples(*[row[j].cur_domain() for row in board]))
            constraints.append(c)
    return constraints


def generate_binary_sat_tuples(dom_x, dom_y, inequality='.'):
    function = None
    if inequality == '.': function = lambda t: t[0] != t[1]
    elif inequality == '>': function = lambda t: t[0] > t[1]
    elif inequality == '<': function = lambda t: t[0] < t[1]
    else: return False
    return filter(function, itertools.product(dom_x, dom_y))


def generate_all_diff_sat_tuples(*doms):
    sat_tuples = []
    n = len(doms)
    for t in itertools.permutations(list(range(1, n+1))):
        in_dom = True
        for i, dom in enumerate(doms):
            if t[i] not in dom:
                in_dom = False
                break
        if in_dom: sat_tuples.append(t)
    return sat_tuples
