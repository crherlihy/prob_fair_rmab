'''RestlessArm: an arm whose states evolve based on an underlying MDP
'''
from typing import Callable
import logging
import numpy as np

from src.Arms.Arm import Arm

class RestlessArm(Arm):
    '''RestlessArm: an arm whose states evolve based on an underlying MDP
    '''
    def __init__(self, 
                 id: int, 
                 transition_matrix: np.ndarray,
                 seed: int = 0, 
                 initial_state: int = 1,
                 n_states: int = None, 
                 n_actions: int = None,
                 rho: Callable = lambda b: b, 
                 r: Callable = None, 
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False, 
                 **kwargs):
        '''
        Init a RestlessArm, which transitions states according to the MDP transition_matrix
        
        :param id: ID of the arm
        :param transition_matrix: (n_actions x n_states x n_states) np.ndarray
        :param seed: seeds the rng, default 0
        :param initial_state: initial state of the arm, defaults to 1. If -1, random
        :param n_states: int number of states, default 2
        :param n_actions: int number of actions, default 2
        :param rho: Callable, optional local reward rho(b) function for WhittleIndexPolicy
        :param r: Callable, optional local reward r(s) function for computing local reward
        :param error_log: error log, defaults to logging.getLogger('error_log')
        :param verbose: flag for extra prints to the console, defaults to False
        :param **kwargs: optional kwargs, passed into Arm

        '''
        
        # Transition matrix is a required kwarg, has a required shape:
        assert(len(transition_matrix.shape)==3)
        assert(transition_matrix.shape[1]==transition_matrix.shape[2])
        if n_states is None:
            n_states = transition_matrix.shape[1]
        else:
            assert(n_states == transition_matrix.shape[1])
        if n_actions is None:
            n_actions = transition_matrix.shape[0]
        else:
            assert(n_actions == transition_matrix.shape[0])
        
        # Initialize general arm properties
        Arm.__init__(self, id=id, error_log=error_log, verbose=verbose, **kwargs)
        self.transition = transition_matrix
        self.n_states = n_states
        self.n_actions = n_actions
        self.initial_state = initial_state
        self.reset(seed=seed, initial_state=initial_state)
        
        # Initialize reward function properties
        self.rho = rho # rho(b) is used to calculate Whittle index
        if r is None:
            self.r = rho # r(s) is used to calculate local reward
        else: 
            self.r = r

    def reset(self, seed: int, initial_state: int = None):
        '''
        Resets in place:
            self.rng
            self.actions
            self.state
            self.belief
            self.last_known_state
            self.time_since_pulled
            self.belief_chains
        
        :param seed: seed for the rng Generator
        :param initial_state: int initial states, defaults to self.initial_state
        
        Simulations should pass in a seed to standardize transitions.
        '''
        
        self.rng = np.random.default_rng(seed)
        
        self.actions = []
        
        if initial_state is None:
            initial_state = self.initial_state
            
        if initial_state == -1:
            self.state = [self.rng.integers(self.n_states)]
        else: 
            self.state = [initial_state]
        
        self.belief = self.state.copy()
        
        self.last_known_state = 1 # Used for whittle index computation
        self.time_since_pulled = 1  # Time since pulled = # of days since the arm was last pulled. If pulled in t, it gets updated to = 1 at t+1.

        # gets initialized/computed when WhittleIndex policy is called
        self.belief_chains = None
        
    def compute_next_belief(self, action: int):
        '''
        Belief is recursive. Given the previous belief and the action computed, return the next belief.
        
        :param action: int in arange(n_actions)
        :return: the next belief given action

        '''
        return self.belief[-1] * self.transition[action, 1, 1] + \
                      (1 - self.belief[-1]) * self.transition[action, 0, 1]
        
    def _update_true_state(self, action: int):
        '''
        Updates (appends) in place:
            the (true) self.state of the arm based on the action
            self.actions
        Must be called before self._update_belief_state. See self.update()
        
        :param action: int in arange(n_actions)
        
        The action at index t represents the action taken when going from state t to t+1
        e.g. actions[1] is the action that moves state[1] to state[2]
        '''
        self.actions.append(action)

        # This update applies to Restless AND Collapsing arms
        if action == 0:
            self.time_since_pulled +=1
        else:
            self.time_since_pulled = 1
        
        outcome = self.rng.random()
        if outcome <= self.transition[action, self.state[-1], 0]: 
            self.state.append(0)
        else:
            self.state.append(1)

        if type(self).__name__ == "RestlessArm":
            self.last_known_state = self.state[-1]
            # Collapsing arm's "last_known_state" gets in self._update_belief_state()
        return


    def _update_belief_state(self):
        '''
        Updates (appends) in place:
            self.belief with self.state
        Must be called after self._update_true_state(), see self.update()

        '''
        if len(self.belief) != len(self.state) - 1:
            raise ValueError('Ensure self._update_true_state() has been called first')
        self.belief.append(self.state[-1])

    def update(self, action: int):
        '''
        Updates self.arm when action is taken
        
        :param action: int in arange(n_actions)

        '''
        self._update_true_state(action=action)
        self._update_belief_state()

    def compute_rho(self, t: int):
        '''
        Calculate local reward of belief, rho(b), at t
        
        :param t: int timestep 
        :return: rho(b_t), float

        '''
        return self.rho(self.belief[t])
    
    def compute_r(self, t: int):
        '''
        Calculate local reward of true state, r(s), at t
        
        :param t: int timestep
        :return: r(s_t), float

        '''
        return self.r(self.state[t])


if __name__ == "__main__":
    pass
