import sys
import time
import random


"""
COMP3702 Tutorial 6 VI starter template.

Last updated by njc 05/09/23
"""


class GridWorldState:
    """
    Class representing a state in the Tutorial 6 Grid World Environment.
    """

    def __init__(self, row: int, col: int, key_collected: bool):
        self.row = row
        self.col = col
        self.key_collected = key_collected

    def __eq__(self, other):
        if not isinstance(other, GridWorldState):
            return False
        return self.row == other.row and self.col == other.col and self.key_collected == other.key_collected

    def __hash__(self):
        return hash((self.row, self.col, self.key_collected))

    def __repr__(self):
        return f'({self.row}, {self.col}, {self.key_collected})'

    def deepcopy(self):
        return GridWorldState(self.row, self.col, self.key_collected)


class GridworldEnv:
    """
    Class representing a Grid World Environment.
    """

    # Directions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTIONS = [UP, DOWN, LEFT, RIGHT]
    ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    # used by perform_action
    DESIRED = 0
    PERPENDICULAR_CW = 1
    PERPENDICULAR_CCW = 2
    ACTION_MAP = {UP: {DESIRED: UP, PERPENDICULAR_CCW: LEFT, PERPENDICULAR_CW: RIGHT},
                  DOWN: {DESIRED: DOWN, PERPENDICULAR_CCW: RIGHT, PERPENDICULAR_CW: LEFT},
                  LEFT: {DESIRED: LEFT, PERPENDICULAR_CCW: DOWN, PERPENDICULAR_CW: UP},
                  RIGHT: {DESIRED: RIGHT, PERPENDICULAR_CCW: UP, PERPENDICULAR_CW: DOWN}}

    def __init__(self):
        self.n_rows = 3
        self.n_cols = 4
        self.p = 0.8        # probability of movement success
        self.gamma = 0.9    # discount factor

        # start at bottom left corner without key
        self.init_state = GridWorldState(2, 0, False)

        # the agent cannot move into obstacle positions, and will be bounced back to their original position
        self.obstacle_positions = {(1, 1)}      # row, col

        # moving into a hazard position will incur a penalty and end the episode
        self.hazard_positions = {(1, 3)}        # row, col
        self.hazard_penalty = 100.0

        # if key_collected is False, moving into the key position will set key_collected = True
        self.key_position = (2, 2)      # row, col

        # moving into the goal position when key_collected is True will give a reward and end the episode
        self.goal_position = (0, 3)     # row, col
        self.goal_reward = 1.0

        # represent 'exited' state as row = -1, col = -1, key_collected = True
        self.exited_state = GridWorldState(-1, -1, True)
        self.states = list(GridWorldState(r, c, k) for r in range(self.n_rows) for c in range(self.n_cols)
                           for k in [True, False] if (r, c) not in self.obstacle_positions) + [self.exited_state]

    def perform_action(self, state: GridWorldState, action: int) -> tuple[tuple[int, int], float, bool]:
        """
        Perform the given action on the given state, sample an outcome, and return the resulting next state, the reward
        received, and if a terminal condition has been reached.
        :param state: a GridWorldState instance
        :param action: an element of GridWorldEnv.ACTIONS
        :return: (next_state [GridWorldState], reward [float], is_terminal [bool])
        """

        next_state: GridWorldState = GridWorldState(0, 0, 0)
        reward: float = 0
        is_terminal: bool = False

        if state == self.exited_state:
            return state, 0, True

        if (state.row, state.col) in self.hazard_positions:
            return self.exited_state, -1*self.hazard_penalty, True
        elif (state.row, state.col) == self.goal_position and state.key_collected:
            return self.exited_state, self.goal_reward, True

        actual_action = self.ACTION_MAP[action][random.choices(
            [self.DESIRED, self.PERPENDICULAR_CCW, self.PERPENDICULAR_CW],
            weights=[self.p, 0.1, 0.1]
        )[0]]

        next_state = self.get_actual_next_state(state, actual_action)

        return next_state, reward, is_terminal
    
    def get_actual_next_state(self, state: GridWorldState, actual_action: int) -> GridWorldState:
        if actual_action == 0: # UP
            next_pos = (state.row-1, state.col)
        elif actual_action == 1: # DOWN
            next_pos = (state.row+1, state.col)
        elif actual_action == 2: # LEFT
            next_pos = (state.row, state.col-1)
        elif actual_action == 3: # RIGHT
            next_pos = (state.row, state.col+1)

        if state.key_collected or next_pos == self.key_position:
            next_state = GridWorldState(next_pos[0], next_pos[1], True)
        else:
            next_state = GridWorldState(next_pos[0], next_pos[1], False)

        # if next_state not in self.states:
        #     next_state = state
        if not((0 <= next_pos[0] < self.n_rows) and (0 <= next_pos[1] < self.n_cols)) or next_pos in self.obstacle_positions:
            next_state = state


        return next_state

    def get_transition_outcomes(self, state: GridWorldState, action: int):
        """
        Return a list of (probability, next_state, reward) tuples representing each possible outcome of performing the
        given action from the given state.
        :param state: a GridWorldState instance
        :param action: an element of GridWorldEnv.ACTIONS
        :return: list of (probability, next_state, reward) tuples
        """
        if state == self.exited_state:
            return [(1, state, 0)]

        if (state.row, state.col) in self.hazard_positions:
            return [(1, self.exited_state, -1*self.hazard_penalty)]
        if (state.row, state.col) == self.goal_position and state.key_collected:
            return [(1, self.exited_state, self.goal_reward)]

        outcomes: list[tuple[float, GridWorldState, float]] = []

        for i in [self.DESIRED, self.PERPENDICULAR_CW, self.PERPENDICULAR_CCW]:
            actual_action = self.ACTION_MAP[action][i]
            probability = self.p if actual_action == action else (1-self.p) / 2.0
            next_state = self.get_actual_next_state(state, actual_action)

            outcomes.append((probability, next_state, 0))

        return outcomes

    def render(self, state):
        """
        Print a text representation of the given state to stdout
        :param state: State to render
        """
        output = ''
        for r in range(self.n_rows):
            line = ''
            for c in range(self.n_cols):
                if r == state.row and c == state.col:
                    glyph = 'A'     # agent
                elif (r, c) in self.obstacle_positions:
                    glyph = 'O'     # obstacle
                elif (r, c) in self.hazard_positions:
                    glyph = '!'     # hazard
                elif (r, c) == self.key_position and not state.key_collected:
                    glyph = 'k'     # key
                elif (r, c) == self.goal_position:
                    glyph = '*'     # goal
                else:
                    glyph = ' '
                line += f'[{glyph}]'
            output += f'{line}\n'
        print(output)


class VISolver:

    EPSILON = 0.001
    MAX_ITER = 100

    VERBOSE = False

    def __init__(self, env: GridworldEnv):
        self.env: GridworldEnv = env
        self.values = {state: 0 for state in self.env.states}
        self.policy = dict()
        self.is_converged: bool = False

        #
        # TODO: add any additional variables and initialization you require here
        #

        pass

    def vi_iteration(self):
        """
        Perform a single iteration of VI.
        """
        new_values: dict = {}
        new_policy: dict = {}
        for state in self.env.states:
            value = -float('inf')
            besta = None
            for a in self.env.ACTIONS:
                ts: list[tuple] = self.env.get_transition_outcomes(state, a)
                Q = sum([
                    t[0] * (t[2] + self.env.gamma * self.values[t[1]])
                    for t in ts
                ])
                if Q > value:
                    value = Q
                    besta = a
            new_values[state] = value
            new_policy[state] = besta

        diffs = [
            abs(self.values[s] - new_values[s]) for s in self.env.states
        ]
        if max(diffs) < self.EPSILON:
            self.is_converged = True

        self.values = new_values
        self.policy = new_policy

    def vi_is_converged(self):
        """
        Return true if VI has converged.
        :return: Ture if converged, False otherwise
        """

        return self.is_converged

    def vi_plan_offline(self):
        """
        Plan an optimal policy using Value Iteration.
        """
        # if self.VERBOSE:
        #     print('Initial values:')
        #     self.print_values()
        for i in range(self.MAX_ITER):
            self.vi_iteration()
            # if self.VERBOSE:
            #     print(f'Values after iteration {i + 1}:')
            #     self.print_values()
            #     print('')   # blank line
            if self.vi_is_converged():
                if self.VERBOSE:
                    print(f'Values converged after {i + 1} iterations!')
                break

    def vi_select_action(self, state):
        """
        Select the optimal action for the given state based on the stored values. You may assume that vi_plan_offline
        has been called before the first time this method is called.
        :param state: a GridWorldState instance
        :return: the optimal action to perform, an element of GridWorldEnv.ACTIONS
        """
        return self.policy[state]

    def print_values(self):
        """
        Print state: value for every state in the state space.
        """
        for state, value in self.values.items():
            print(state, round(value, 4), end=';  ')


def main(arglist):
    env = GridworldEnv()
    solver = VISolver(env)

    # run value iteration
    t0 = time.time()
    solver.vi_plan_offline()
    runtime = time.time() - t0
    print(f'Time to complete: {runtime} seconds')

    # simulate an episode
    r_total = 0.0
    s = env.init_state
    env.render(s)
    while True:
        a = solver.vi_select_action(s)
        # print(f'Selected action: {env.ACTION_NAMES[a]}')
        s1, r, is_terminal = env.perform_action(s, a)
        r_total += r
        # env.render(s1)
        s = s1
        if is_terminal:
            break
    print(f'Episode completed with total reward {r_total}!')


if __name__ == '__main__':
    main(sys.argv[1:])
