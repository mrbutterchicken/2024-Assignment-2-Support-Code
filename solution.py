import sys
from constants import *
from environment import *
from state import State
import numpy as np
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

"""

class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        self._states: list[State] = []
        self._state_index: dict[int, State] = {}
        self._transition_cache: dict[tuple[State, int], list] = {}
        self._solved_states: list[State] = []

        self.vi_values: dict[State, float] = {} # State: Reward
        self._policy: dict[State, int] = {} # State: Action
        self._policy_vector = None

        self._rewards = None

        self._vi_converged: bool = False
        self._pi_converged: bool = False

        self._transition_model: np.ndarray = None

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        return [1, 2, 3, 4, 5, 6]

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

    def get_all_states(self, starting: State) -> None:
        init = starting
        frontier: list[State] = [init]
        visited: list[State] = [init]

        while len(frontier):
            state = frontier.pop()

            for child in self.expand(state):
                if child not in visited:
                    visited.append(child)
                    frontier.append(child)

        # visited.reverse()
        self._states = visited
        # self._states.append(self.environment.exited_state)

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.get_all_states(starting=self.environment.get_init_state())

        self.vi_values = {state: 0 for state in self._states}
        self._policy = {state: BEE_ACTIONS[0] for state in self._states}

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
            if self.environment.is_solved(state):
                continue
            value = -float('inf')
            action = None
            for a in BEE_ACTIONS:
                ts: list[tuple] = self.get_transition_outcomes(state, a)
                Q = sum([
                    t[0] * (t[2] + (self.environment.gamma) * self.vi_values[t[1]])
                    for t in ts
                ])
                if Q > value:
                    value = Q
                    action = a
            if abs(value - self.vi_values[state]) > maxdiff:
                maxdiff = abs(value - self.vi_values[state])
            self.vi_values[state] = value
            self._policy[state] = action

        if maxdiff <= self.environment.epsilon:
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
        return self.vi_values[state]

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self._policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        self.get_all_states(starting=self.environment.get_init_state())
        self._policy_vector = np.zeros([len(self._states)], dtype=np.int64)
        self._state_index = {s: i for i, s in enumerate(self._states)}

        # ----- TRANSITION MODEL: Prob(State[i] x Action[j] ==> State[k]) -----
        self._transition_model = np.zeros([
            len(self._states), 
            len(BEE_ACTIONS), 
            len(self._states)
        ])
        for i, s in enumerate(self._states):
            for j, a in enumerate(BEE_ACTIONS):
                for prob, next_state, _ in self.get_transition_outcomes(s, a):
                    self._transition_model[i][j][self._state_index[next_state]] += prob

        # --------- REWARD MATRIX MODEL: Reward(State[i] x Action[j]) ---------
        self._rewards = np.zeros([
            len(self._states),
            len(BEE_ACTIONS)
        ])
        for i, s in enumerate(self._states):
            for j, a in enumerate(BEE_ACTIONS):
                self._rewards[i][j] = self.get_expected_reward(s, a)

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self._pi_converged

    def policy_evaluation(self) -> dict[State, float]:
        state_numbers = np.array(range(len(self._states)))

        P = self._transition_model[state_numbers, self._policy_vector]
        rewards = self._rewards[state_numbers, self._policy_vector]

        values = np.linalg.solve(
            np.identity(len(self._states)) - (self.environment.gamma * P), rewards
        )
        return {state: values[i] for i, state in enumerate(self._states)}

    def policy_improvement(self, values: dict[State, float]) -> None:
        changed = False
        for i, state in enumerate(self._states):
            value = -float('inf')
            action = None
            for a in BEE_ACTIONS:
                ts: list[tuple] = self.get_transition_outcomes(state, a)
                Q = sum([
                    prob * (reward + self.environment.gamma * values[next_state])
                    for prob, next_state, reward in ts
                ])
                if Q > value:
                    value = Q
                    action = a
            if action != self._policy_vector[i]:
                self._policy_vector[i] = action
                changed = True

        if not changed:
            self._pi_converged = True

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        V_pi = self.policy_evaluation()
        self.policy_improvement(V_pi)

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
        return self._policy_vector[self._state_index[state]]

    # === Helper Methods ===============================================================================================

    def get_transition_outcomes(self, state: State, action: int) -> list[tuple[float, State, float]]:

        if self.environment.is_solved(state):
            return [(1, state, 0)]

        if (state, action) in self._transition_cache:
            return self._transition_cache[(state, action)]

        outcomes: list[tuple[float, State, float]] = list()

        D = self.environment.double_move_probs[action]
        C = self.environment.drift_cw_probs[action]
        CC = self.environment.drift_ccw_probs[action]

        c_drift = C * (1-D) # * (1-CC)
        cc_drift = CC * (1-D) # * (1-C)
        double_only = D * (1-C) * (1-CC)
        c_d_dub = C*D # * (1-CC)
        cc_d_dub = CC*D # * (1-C)
        desired = 1 - (c_drift + cc_drift + double_only + c_d_dub + cc_d_dub)

        movements: list[tuple[float, list[int]]] = [
            (desired, [action]),
            (c_drift, [SPIN_RIGHT, action]),
            (cc_drift, [SPIN_LEFT, action]),
            (double_only, [action, action]),
            (c_d_dub, [SPIN_RIGHT, action, action]),
            (cc_d_dub, [SPIN_LEFT, action, action])
        ]

        for prob, moves in movements:
            min_reward = 0
            new_state = state
            for m in moves:
                reward, new_state = self.environment.apply_dynamics(new_state, m)
                # use the minimum reward over all movements
                if reward < min_reward:
                    min_reward = reward
            outcomes.append((prob, new_state, min_reward))

        self._transition_cache[(state, action)] = outcomes
        return outcomes

    def get_expected_reward(self, state: State, action: int) -> float:
        return sum(
            [p*r for p, _, r in self.get_transition_outcomes(state, action)]
        )
