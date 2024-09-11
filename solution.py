import sys
import time
from constants import *
from environment import *
from state import State
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

"""

class Node:
    def __init__(self, state: State, action_from_parent: int, parent) -> None:
        self.state = state
        self.action_from_parent = action_from_parent
        self.parent = parent

    # def __str__(self) -> str:
    #     return str((self.cost, self.state))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return (self.state == other.state)
        else:
            return False

class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        self._states: list[State] = []
        self.vi_values: dict[State, float] = {} # State: Reward
        self.vi_policy: dict[State, int] = {} # State: Action
        self._vi_converged: bool = False
        self._pi_converged: bool = False

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        return [1, 2, 4]

    # === Value Iteration ==============================================================================================

    def expand(self, state: State) -> list[State]:
        expanded: list[State] = []
        for action in BEE_ACTIONS:
            _, next_state = self.environment.apply_dynamics(
                state,
                action
            )
            if next_state != state:
                expanded.append(next_state)
        return expanded

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self._states = list()
        init = self.environment.get_init_state()
        frontier: list[State] = [init]
        visited: list[State] = list()
        while len(frontier):
            state = frontier.pop()

            for child in self.expand(state):
                if child not in visited:
                    visited.append(child)
                    frontier.append(child)

        visited.reverse()
        self._states = visited

        self.vi_values = {state: 0 for state in self._states}
        self.vi_policy = {state: BEE_ACTIONS[0] for state in self._states}

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self._vi_converged

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        maxdiff = 0
        for state in self._states:
            value = -float('inf')
            action = None
            for a in BEE_ACTIONS:
                ts: list[tuple] = self.get_transition_outcomes(state, a)
                Q = sum([
                    t[0] * (t[2] + self.environment.gamma * self.vi_values.get(t[1], 0))
                    for t in ts
                ])
                if Q > value:
                    value = Q
                    action = a
            if abs(value - self.vi_values[state]) > maxdiff:
                maxdiff = abs(value - self.vi_values[state])
            self.vi_values[state] = value
            self.vi_policy[state] = action

        if maxdiff < self.environment.epsilon:
            self._vi_converged = True



    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.vi_values.get(state, 0)

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.vi_policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        #
        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Helper Methods ===============================================================================================

    def get_transition_outcomes(self, state: State, action: int) -> list[tuple[float, State, float]]:

        if state == self.environment.exited_state:
            return [(1, self.environment.exited_state, 0)]
        if self.environment.is_solved(state):
            return [(1, self.environment.exited_state, -self.environment.reward_tgt)]

        outcomes: list[tuple[float, State, float]] = list()

        D = self.environment.double_move_probs[action]
        C = self.environment.drift_cw_probs[action]
        CC = self.environment.drift_ccw_probs[action]

        desired = 1 - (C + CC + D - D*C - D*CC + C*CC*D)
        c_drift = C-C*D
        cc_drift = CC-CC*D
        double_only = D - C*D - CC*D + C*CC*D
        c_d_dub = C*D
        cc_d_dub = CC*D

        movements: dict[float, list[int]] = {
            desired: [action],
            c_drift: [SPIN_CW, action],
            cc_drift: [SPIN_CCW, action],
            double_only: [action, action],
            c_d_dub: [SPIN_CW, action, action],
            cc_d_dub: [SPIN_CCW, action, action]
        }

        for prob in movements:
            min_reward = 0
            new_state = state
            if prob > 0:
                for m in movements[prob]:
                    reward, new_state = self.environment.apply_dynamics(new_state, m)
                    # use the minimum reward over all movements
                    if reward < min_reward:
                        min_reward = reward
                outcomes.append((prob, new_state, min_reward))

        return outcomes

