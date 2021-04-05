# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


"""
Skeleton code for Project 1 of Columbia University's AI EdX course (8-puzzle).
Python 3
"""

# import queue as Q

import time

import sys

from collections import deque

if sys.platform == "win32":
    import psutil
    # print("psutil", psutil.Process().memory_info().rss)
else:
    # Note: if you execute Python from cygwin,
    # the sys.platform is "cygwin"
    # the grading system's sys.platform is "linux2"
    import resource
    # print("resource", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


import math



#### SKELETON CODE ####

## The Class that Represents the Puzzle

class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n*n != len(config) or n < 2:

            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = i // self.n

                self.blank_col = i % self.n

                break

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print(line)

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:
                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:
                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:
                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:
                self.children.append(right_child)

        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters


def writeOutput(statisits):
    path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage = statisits
    
    f = open('output.txt', 'w')
    f.write(f'path_to_goal: {path_to_goal}\n')
    f.write(f'cost_of_path: {cost_of_path}\n')
    f.write(f'nodes_expanded: {nodes_expanded}\n')
    f.write(f'search_depth: {search_depth}\n')
    f.write(f'max_search_depth: {max_search_depth}\n')
    f.write(f'running_time: {running_time}\n')
    f.write(f'max_ram_usage: {max_ram_usage}')
    f.close()


def bfs_search(initial_state):

    """BFS search"""

    ### STUDENT CODE GOES HERE ###

    # assert isinstance(initial_state, PuzzleState), 'Input does not match the specification. '


    frontier = deque([initial_state])
    explored = set(frontier)

    nodes_expanded = 0
    max_search_depth = 0
    # set up a bookkeeping dictionary to backtrack solution once found
    solution_track = {}# key: current state; value: (action, parent state)

    while frontier:

        # print('Number of expanded nodes', nodes_expanded)

        state = frontier.popleft()
        explored.add(state.config)

        # print('Frontier', [item.config for item in frontier])
        # print('Explored set',  [item.config for item in explored])

        if test_goal(state):
            path_to_goal = []
            cur_state = state
            cost_of_path = state.cost
            search_depth = state.cost
            # backtracking process to retrieve solution
            while True:

                prev_state = solution_track[cur_state]
                path_to_goal.append(cur_state.action)

                if prev_state.config == initial_state.config:
                    break
                else:
                    cur_state = prev_state

            path_to_goal.reverse()


            return path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth+1

        for neighbor in state.expand():

            if neighbor.config not in explored:

                frontier.append(neighbor)
                explored.add(neighbor.config)
                solution_track[neighbor] = state

        max_search_depth = max(max_search_depth, state.cost)
        nodes_expanded += 1







def dfs_search(initial_state):

    """DFS search"""

    ### STUDENT CODE GOES HERE ###
    # assert isinstance(initial_state, PuzzleState), 'Input does not match the specification. '

    frontier = deque([initial_state]); frontier_set = {initial_state.config}
    explored = set(frontier_set)

    nodes_expanded = 0
    max_search_depth = 0
    # set up a bookkeeping dictionary to backtrack solution once found
    solution_track = {}  # key: current state; value: (action, parent state)

    while frontier:

        if nodes_expanded % 1000 == 0:
            print('Number of expanded nodes', nodes_expanded)

        state = frontier.pop()
        explored.add(state.config)

        if test_goal(state):
            path_to_goal = []
            cur_state = state
            cost_of_path = state.cost
            search_depth = state.cost

            # backtracking process to retrieve solution
            while True:

                prev_state = solution_track[cur_state]
                path_to_goal.append(cur_state.action)

                if prev_state.config == initial_state.config:
                    break
                else:
                    cur_state = prev_state

            path_to_goal.reverse()

            return path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth

        for neighbor in state.expand()[::-1]:

            if neighbor.config not in explored:

                frontier.append(neighbor)
                frontier_set.add(neighbor.config)
                explored.add(neighbor.config)
                solution_track[neighbor] = state

        max_search_depth = max(max_search_depth, state.cost)
        nodes_expanded += 1


def A_star_search(initial_state):

    """A * search"""

    ### STUDENT CODE GOES HERE ###

    # assert isinstance(initial_state, PuzzleState), 'Input does not match the specification. '

    frontier = [initial_state]; frontier_set={initial_state.config}
    explored_set = {initial_state.config}
    nodes_expanded = 0
    max_search_depth = 0

    # set up a bookkeeping dictionary to backtrack solution once found
    solution_track = {}  # key: current state; value: (action, parent state)

    while frontier:

        # print('Number of expanded nodes', nodes_expanded)

        frontier = sorted(frontier, key=calculate_total_cost)
        state = frontier.pop(0)
        explored_set.add(state.config)

        if test_goal(state):
            path_to_goal = []
            cur_state = state
            cost_of_path = state.cost
            search_depth = state.cost

            # backtracking process to retrieve solution
            while True:

                prev_state = solution_track[cur_state]
                path_to_goal.append(cur_state.action)

                if prev_state.config == initial_state.config:
                    break
                else:
                    cur_state = prev_state

            path_to_goal.reverse()

            return path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth+1

        for neighbor in state.expand():

            if neighbor.config not in explored_set:
                frontier.append(neighbor) # we would heapify the list at the beginning of the loop
                frontier_set.add(neighbor.config)
                solution_track[neighbor] = state

            elif neighbor.config in frontier_set:
                if neighbor in solution_track.keys() and calculate_total_cost(neighbor) >= 1 + calculate_total_cost(neighbor):
                    solution_track[neighbor] = state

        max_search_depth = max(max_search_depth, state.cost)
        nodes_expanded += 1


def calculate_total_cost(state):

    """calculate the total estimated cost of a state"""

    ### STUDENT CODE GOES HERE ###
    # assert isinstance(state, PuzzleState)

    heuristic_cost = 0
    # First calculate the number of misplaced tiles,
    cur_config = state.config

    for ix, tile in enumerate(cur_config):
        # heuristic cost
        heuristic_cost += calculate_manhattan_dist(ix, tile, state.n)

    return state.cost +  heuristic_cost

def calculate_manhattan_dist(idx, value, n):

    """calculate the manhattan distance of a tile"""

    ### STUDENT CODE GOES HERE ###
    if not value:
        return 0

    cur_row_idx = idx // n + 1
    cur_col_idx = idx % n + 1

    goal_row_idx = value // n + 1
    goal_col_idx = value % n + 1

    return abs(cur_row_idx - goal_row_idx) + abs(cur_col_idx - goal_col_idx)

def test_goal(puzzle_state):

    """test the state is the goal state or not"""

    ### STUDENT CODE GOES HERE ###

    # assert isinstance(puzzle_state, PuzzleState), 'Input does not match the specification. '

    return puzzle_state.config == tuple(i for i in range(puzzle_state.n**2))


# Main Function that reads in Input and Runs corresponding Algorithm

def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))  # (3,1,2,0,4,5,6,7,8)

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":
        
        start = time.time()
        result = bfs_search(hard_state)
        end = time.time()
        memory = psutil.Process().memory_info().rss
        elapsed_time = round(end-start, 8)
        
        result = [*result, elapsed_time, memory]
        if result is not None:

            writeOutput(result)

    elif sm == "dfs":

        start = time.time()
        result = dfs_search(hard_state)
        end = time.time()
        memory = psutil.Process().memory_info().rss
        elapsed_time = round(end - start, 8)

        result = [*result, elapsed_time, memory]
        if result is not None:
            writeOutput(result)

    elif sm == "ast":

        start = time.time()
        result = A_star_search(hard_state)
        end = time.time()
        memory = psutil.Process().memory_info().rss
        elapsed_time = round(end - start, 8)

        result = [*result, elapsed_time, memory]
        if result is not None:
            writeOutput(result)

    else:

        print("Enter valid command arguments !")


if __name__ == '__main__':

    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
