from __future__ import nested_scopes
from checkers_game import *

# MINIMAX: cache[(color, limit, board_tuple)] = value
# ALPHABETA: cache[(color, limit, board_tuple)] = {flag: (-1, 0, 1), value}
cache = {} # you can use this to implement state caching!

# Method to compute utility value of terminal state
def compute_utility(state, color):
    # IMPLEMENT
    r_count, b_count = 0, 0
    for i, row in enumerate(state.board):
        for cell in row:
            if cell == '.': continue
            elif cell == 'r': r_count += 1
            elif cell == 'R': r_count += 3
            elif cell == 'b': b_count += 1
            elif cell == 'B': b_count += 3
            # elif cell == 'r': r_count += 5 + 2*(i<4)
            # elif cell == 'R': r_count += 10
            # elif cell == 'b': b_count += 5 + 2*(i>=4)
            # elif cell == 'B': b_count += 10
    if color == 'r': return r_count - b_count
    elif color == 'b': return b_count - r_count
    return False


# Better heuristic value of board
def compute_heuristic(state, color): 
    # IMPLEMENT
    r_count, b_count = 0, 0
    for i, row in enumerate(state.board):
        for cell in row:
            if cell == '.': continue
            elif cell == 'r': r_count += 1
            elif cell == 'R': r_count += 3
            elif cell == 'b': b_count += 1
            elif cell == 'B': b_count += 3
            # elif cell == 'r': r_count += 5 + 2*(i<4)
            # elif cell == 'R': r_count += 10
            # elif cell == 'b': b_count += 5 + 2*(i>=4)
            # elif cell == 'B': b_count += 10
    if color == 'r': return r_count - b_count
    elif color == 'b': return b_count - r_count
    return False


def change_color(color):
    if color == 'r': return 'b'
    elif color == 'b': return 'r'
    return False


############ MINIMAX ###############################
def minimax_min_node(state, color, limit, caching=0):
    # IMPLEMENT
    if caching and (color, limit, tuple(tuple(row) for row in state.board)) in cache:
        return cache[(color, limit, tuple(tuple(row) for row in state.board))], Board(None, None)
    succs = successors(state, color)
    if limit == 0 or not succs:
        return compute_utility(state, change_color(color)), Board(None, None)
    value, best_move = float('inf'), Board(None, None)
    for succ in succs:
        next_val, next_move = minimax_max_node(succ, change_color(color), limit-1, caching)
        if caching:
            cache[(change_color(color), limit-1, tuple(tuple(row) for row in succ.board))] = next_val
        if next_val < value:
            value, best_move = next_val, succ
    return value, best_move


def minimax_max_node(state, color, limit, caching=0):
    # IMPLEMENT
    if caching and (color, limit, tuple(tuple(row) for row in state.board)) in cache:
        return cache[(color, limit, tuple(tuple(row) for row in state.board))], Board(None, None)
    succs = successors(state, color)
    if limit == 0 or not succs:
        return compute_utility(state, color), Board(None, None)
    value, best_move = float('-inf'), Board(None, None)
    for succ in succs:
        next_val, next_move = minimax_min_node(succ, change_color(color), limit-1, caching)
        if caching:
            cache[(change_color(color), limit-1, tuple(tuple(row) for row in succ.board))] = next_val
        if next_val > value:
            value, best_move = next_val, succ
    return value, best_move


def select_move_minimax(state, color, limit, caching=0):
    """
        Given a state (of type Board) and a player color, decide on a move.
        The return value is a list of tuples [(i1,j1), (i2,j2)], where
        i1, j1 is the starting position of the piece to move
        and i2, j2 its destination.  Note that moves involving jumps will contain
        additional tuples.

        Note that other parameters are accepted by this function:
        If limit is a positive integer, your code should enforce a depth limit that is equal to the value of the parameter.
        Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
        value (see compute_utility)
        If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
        If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    """
    # IMPLEMENT
    cache.clear()
    value, best_move = minimax_max_node(state, color, limit, caching)
    return best_move.move


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(state, color, alpha, beta, limit, caching=0, ordering=0):
    # IMPLEMENT
    beta_orig = beta
    if caching and (color, limit, tuple(tuple(row) for row in state.board)) in cache:
        cached = cache[(color, limit, tuple(tuple(row) for row in state.board))]
        if cached['flag'] < 0:
            if cached['value'] > alpha: alpha = cached['value']
        elif cached['value'] < 0:
            if cached['value'] < beta: beta = cached['value']
        else: return cached['value'], Board(None, None)
        if alpha >= beta: return cached['value'], Board(None, None)
    succs = successors(state, color)
    if limit == 0 or not succs:
        return compute_utility(state, change_color(color)), Board(None, None)
    if ordering:
        succs.sort(key=lambda x: compute_heuristic(x, color), reverse=True)
    value, best_move = float('inf'), Board(None, None)
    for succ in succs:
        next_val, next_move = alphabeta_max_node(succ, change_color(color), alpha, beta, limit-1, caching, ordering)
        if next_val < value:
            value, best_move = next_val, succ
            if value < beta: beta = value
            if alpha >= beta: break
    if caching:
        flag = None
        if value <= alpha: flag = 1
        elif value >= beta_orig: flag = -1
        else: flag = 0
        cache[(color, limit, tuple(tuple(row) for row in state.board))] = {'flag': flag, 'value': value}
    return value, best_move


def alphabeta_max_node(state, color, alpha, beta, limit, caching=0, ordering=0):
    # IMPLEMENT
    alpha_orig = alpha
    if caching and (color, limit, tuple(tuple(row) for row in state.board)) in cache:
        cached = cache[(color, limit, tuple(tuple(row) for row in state.board))]
        if cached['flag'] < 0:
            if cached['value'] > alpha: alpha = cached['value']
        elif cached['value'] < 0:
            if cached['value'] < beta: beta = cached['value']
        else: return cached['value'], Board(None, None)
        if alpha >= beta: return cached['value'], Board(None, None)
    succs = successors(state, color)
    if limit == 0 or not succs:
        return compute_utility(state, color), Board(None, None)
    if ordering:
        succs.sort(key=lambda x: compute_heuristic(x, color), reverse=True)
    value, best_move = float('-inf'), Board(None, None)
    for succ in succs:
        next_val, next_move = alphabeta_min_node(succ, change_color(color), alpha, beta, limit-1, caching, ordering)
        if next_val > value:
            value, best_move = next_val, succ
            if value > alpha: alpha = value
            if alpha >= beta: break
    if caching:
        flag = None
        if value <= alpha_orig: flag = 1
        elif value >= beta: flag = -1
        else: flag = 0
        cache[(color, limit, tuple(tuple(row) for row in state.board))] = {'flag': flag, 'value': value}
    return value, best_move


def select_move_alphabeta(state, color, limit, caching=0, ordering=0):
    """
    Given a state (of type Board) and a player color, decide on a move. 
    The return value is a list of tuples [(i1,j1), (i2,j2)], where
    i1, j1 is the starting position of the piece to move
    and i2, j2 its destination.  Note that moves involving jumps will contain
    additional tuples.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enforce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    # IMPLEMENT
    cache.clear()
    value, best_move = alphabeta_max_node(state, color, float('-inf'), float('inf'), limit, caching, ordering)
    return best_move.move


# ======================== Class GameEngine =======================================
class GameEngine:
    def __init__(self, str_name):
        self.str = str_name

    def __str__(self):
        return self.str

    # The return value should be a move that is denoted by a list
    def nextMove(self, state, alphabeta, limit, caching, ordering):
        global PLAYER
        PLAYER = self.str
        if alphabeta:
            result = select_move_alphabeta(Board(state), PLAYER, limit, caching, ordering)
        else:
            result = select_move_minimax(Board(state), PLAYER, limit, caching)

        return result
