from random import randint

from BaseAI import BaseAI



import time

import numpy as np

timeLimit = 0.2
tolerence = 1 # at most eat up tolerence*100% of the time limit
depth_limit = 5
class PlayerAI(BaseAI):
    """
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT"
    """
    def __init__(self):
        self.prevTime = None

    def getMove(self, grid):

        # Simple random move
        # return self._randomMove(grid)
        self.prevTime = time.process_time()
        action = self._minmax(grid)

        # minmax startegy
        if action is not None:
            return action
        else:
            return np.random.choice(range(grid.size))
        
    def _minmax(self, grid):
        cur_depth = 0
        child, _, action = self._maximize(grid, -np.inf, np.inf, cur_depth)
        
        return action

    def _maximize(self, state, alpha, beta, depth):

        # terminal state
        if not state.canMove():
            # print("###### Max node is terminated.")
            return None, state.getMaxTile(), None

        # heuristically terminal state
        if depth > depth_limit or (time.process_time() - self.prevTime) > tolerence * timeLimit :
            return None, self._eval(state), None

        maxChild, maxUtility, maxAction = None, -np.inf, None

        for (action, child) in self._getChildrens(state, method = 'max'):
            _, utility = self._minimize(child, alpha, beta, depth+1)

            if utility > maxUtility:
                maxChild, maxUtility, maxAction = child, utility, action

            if maxUtility >= beta:
                break

            if maxUtility >= alpha:
                alpha = maxUtility

        return maxChild, maxUtility, maxAction


    def _minimize(self, state, alpha, beta, depth):

        # Terminal state
        if not state.getAvailableCells():
            return None, state.getMaxTile()

        if (time.process_time() - self.prevTime) > tolerence * timeLimit or depth > depth_limit:

            return None, self._eval(state)

        minChild, minUtility = None, np.inf

        for child in self._getChildrens(state, method='min'):
            _, utility, _action = self._maximize(child, alpha, beta, depth+1)

            if utility < minUtility:
                minChild, minUtility = child, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility

    def _eval(self, state):

        score_config = np.array(state.map)
        score_config_filter_positive = score_config[score_config > 0]

        weight_mat_snaked = np.array([[2**28, 2**24, 2**20, 2**16],
                               [2**8,  2**9,  2**10, 2**11],
                               [2**7,  2**6,  2**5,  2**4],
                               [2**0,  2**1,  2**2,  2**3]]).astype(np.int64)


        h1 = len(state.getAvailableCells())
        h2 = np.sum(np.multiply( weight_mat_snaked, score_config))

        return h2

    def _getChildrens(self, state, method):

        if method == 'max':

            action_children_pair = []

            for move in state.getAvailableMoves():
                grid = state.clone()
                grid.move(move)
                action_children_pair.append((move, grid))

            return sorted(action_children_pair, key = lambda x: self._eval(x[1]), reverse=True)

        elif method == 'min':

            children = []

            for move in state.getAvailableCells():
                if state.canInsert(move):
                    #for newValue in [2, 4]:
                    grid = state.clone()
                    newValue = np.random.choice([2, 4])
                    grid.insertTile(move, newValue)
                    children.append(grid)

            return sorted(children, key=self._eval, reverse=False)

        else:
            raise Exception(f'{method} unsupported.')



    def _randomMove(self, grid):

        moves = grid.getAvailableMoves()

        return moves[randint(0, len(moves) - 1)] if moves else None