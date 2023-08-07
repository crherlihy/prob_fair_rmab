'''Cohort: Base class for cohorts (collections of Arms)
'''
from typing import Callable
from pydoc import locate
import logging

import numpy as np

class Cohort(object):
    '''Cohort: Base class for cohorts (collections of Arms)
    '''
    def __init__(self, 
                 seed: int, 
                 arm_type: str,
                 n_arms: int,
                 initial_state: int = 1,
                 n_states: int = 2,
                 n_actions: int = 2,
                 pull_action: int = 1,
                 local_reward: str = 'belief_identity',
                 lambd: float = 20.0,
                 rho: Callable = None,
                 r: Callable = None,
                 R: Callable = np.sum,
                 beta: float = 1.0,
                 cohort_name: str = None,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs):
        '''
        
        :param seed: int, seeds self.rng, an np.random.Generator used in the creation of transition matrices and seeding Arms.
        :param arm_type: str of an Arm class
        :param n_arms: int, number of Arms
        :param initial_state: int, -1 indicates random initial state of an Arm, default 1
        :param n_states: int, number of states, defaults to 2
        :param n_actions: int, number of actions, defaults to 2
        :param pull_action: int, defaults to 1
        :param local_reward: str descriptor of local reward function, defaults to 'belief_identity'. Valid values: 'belief_identity', 'mate21_concave'
        :param lambd: float parameter of local reward, defaults to 20.0. Currently only used in local_reward = 'mate21_concave'
        :param rho: Callable, optional local reward rho(b) function
        :param r: Callable, optional local reward r(s) function
        :param R: Callable, optional global reward function R(s), defaults to np.sum
        :param beta: float discount parameter, defaults to 1.0
        :param cohort_name: str name of the cohort
        :param error_log: logging.Logger for errors.
        :param verbose: bool flag for optional console args, defaults to False
        :return: None

        '''
        ## General settings
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.error_log = error_log
        self.arm_type = arm_type # must be the name of a valid Arm class 
        self.n_arms = n_arms
        self.ids = np.arange(n_arms)
        self.initial_state = initial_state # -1 indicates random
        self.n_states = n_states
        self.n_actions = n_actions
        self.pull_action = pull_action
        self.lambd = lambd
        self.verbose = verbose
        self.arm_seeds = self.rng.integers(low=0, high=10**5, size=(self.n_arms))

        ## Reward
        self.local_reward = 'belief_identity' if local_reward in ["None", None] else local_reward
        if rho in [None, 'None']:
            if self.local_reward == 'belief_identity':
                self.rho = lambda b: b
                self.r = self.rho
            elif self.local_reward == 'mate21_concave':
                self.lambd = lambd
                self.rho = lambda b : -np.e**(self.lambd*(1-b))
                self.r = self.rho 
            else:
                # Note: when implementing a local reward function here:
                # make sure corresponding forward/reverse thresh opt. logic is implemented in SyntheticCohort 
                # This will likely require symbolic differentiation to find min/max rho'(b)
                raise ValueError(f'local reward {local_reward} is not understood.')
        elif r in [None, 'None']:
            self.r = rho
        else:
            self.r = r

        self.R = R
        self.beta = beta
        
        if cohort_name is None:
            cohort_name = type(self).__name__
        self.cohort_name = cohort_name
        
    def assign_key(self, *key_count: int, key_label: [str] = None):
        '''
        Randomly assigns a key to each arm id, represents e.g. 
            generation method (forward/reverse threshold optimal)
            demographic information (male/female)
            
        :param *key_count: int, count of arms that should be assigned the same key
        :param key_label: str label for the keys. If None, uses np.arange(len(key_count))
        :return: tuple (keys, key_groups):
            :return keys: list of keys, length self.n_arms
            :return key_groups: list of np.arrays, grouping self.ids by key
        
        Assumes self.ids is np.arange((n_arms)).
        The choice of key label usually matters for classes that inherit from Cohort (see e.g. SyntheticCohort)
        '''
        if key_label is None:
            n_keys = len(key_count)
            key_label = np.arange(n_keys)
        assert(len(key_label)==len(key_count))
        
        ids_unassigned = self.ids
        assert(len(ids_unassigned)==self.n_arms)
        keys = [None]*self.n_arms
        key_groups = [None]*len(key_count)
        
        for i, count in enumerate(key_count):
            ids_selected = self.rng.choice(ids_unassigned, count, replace=False)
            ids_unassigned = np.setxor1d(ids_unassigned, ids_selected)
            for arm_id in ids_selected:
                keys[arm_id] = key_label[i]
            key_groups[i] = ids_selected
        
        return keys, key_groups
        
    def reset_rng(self):
        '''
        Reset self.rng in place. Required for consistent, comparable transition matrix generation.
        '''
        self.rng = np.random.default_rng(self.seed)
        return
    
    def instantiate_arms(self):
        '''
        Instantiate arms
        :return: [RestlessArm]

        '''
        class_ = locate(f'src.Arms.{self.arm_type}.{self.arm_type}')
        arms = [class_(id=arm_id, 
                       transition_matrix=self.transitions[i, ...],
                       seed=self.arm_seeds[i], 
                       initial_state=self.initial_state,
                       n_states=self.n_states, 
                       n_actions=self.n_actions,
                       rho=self.rho, 
                       r=self.r, 
                       error_log=self.error_log,
                       verbose=self.verbose)
                for i, arm_id in enumerate(self.ids)]
        return arms
    
    def reset_arms(self, seeds: [int]):
        '''
        Reset in place the transtitions that actually occur when an action takes place.
        :param seeds: [int] of length self.n_arms
        
        '''
        for i, arm in enumerate(self.arms):
            arm.reset(seed=seeds[i])
    
    def update_arms(self, actions: [int]):
        '''
        Update in place the state, action, and belief of arms when an action takes place.
        :param actions: [int] of length self.n_arms, int must be in arange(n_actions)

        '''
        for i, arm in enumerate(self.arms):
            arm.update(action=actions[i])
        
    def generate_transition_matrices(self):
        '''
        Generate transition matrices
        
        :raises NotImplementedError: implement this method in children classes
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix

        '''
        raise NotImplementedError("This method generate_transition_matrices() " \
                                  "must be implemented in the child class " \
                                  "inheriting from Cohort.")
            
    def generate_random_transition_matrices(self, n_arms: int = None):
        '''
        Generate a random transition matrix.
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
        
        '''
        if n_arms is None:
            n_arms = self.n_arms
        
        transitions = self.rng.uniform(low=0, high=1, 
                                       size=n_arms*self.n_actions*self.n_states*self.n_states) \
                                      .reshape((n_arms, self.n_actions, self.n_states, self.n_states))
        transitions = transitions / np.sum(transitions, axis=-1, keepdims=True)
        return transitions
            
    def validate_transition_matrices(self):
        '''
        Validations performed: dimensions and probabilities
        
        :return: bool True if all assertions evaluate to True.
        '''
        
        assert(self.validate_dimensions())
        assert(self.validate_probabilities())
        return True
    
    def validate_dimensions(self):
        '''
        Validation of self.transitions: shape is n_arms x n_actions x n_states x n_states
        
        :return: True if the dimensions are valid.
    
        '''
        return (np.shape(self.transitions) == (self.n_arms, self.n_actions, self.n_states, self.n_states))
        
    
    def validate_probabilities(self):
        """
        Validation of self.transitions: entries make sense as probabilities.
        
        :return: True if probability entries sum to one and every entry is in [0,1]
    
        """
        summed_entries = np.sum(self.transitions, axis=-1)
        return np.all(np.isclose(summed_entries, 1)) & \
            (np.all((self.transitions >= 0) & (self.transitions <= 1)))
    
    def validate_structural_constraints(self, transitions=None):
        """
        Validation of transitions: relative values of entries satisfy our structural constraints
            (that is, they make sense for interventions)
        
        :return: True if:
            (a) more likely to stay in a good state than to go from bad to good state
            (b) acting is more effective than passivity in going to a good state
    
        """
        if transitions is None:
            transitions = self.transitions
        assert(np.shape(transitions)[-1]==2) # inequalities only supported for two-state system.
        
        # More likely to stay in a good state than go from a bad to good state:
        good_state_inertia = np.all(transitions[...,0,1] < transitions[...,1,1])
        
        # Acting is more effective than being passive in going to a good state:
        intervention_benefit = np.all(transitions[...,0,:,1] < transitions[...,1,:,1])
        
        return good_state_inertia & intervention_benefit
    
    def compute_true_reward(self, t: int = None):
        '''
        Compute global reward.
        
        :param t: timestep, defaults to simulation horizon (T) if None
        :type t: int, optional
        :return: global reward
        :rtype: float

        '''
        if t is None:
            T = len(self.arms[0].state)
            if self.verbose:
                print(f'Computing reward over {T} timesteps.')
            return self.R([self.beta**t*arm.compute_r(t) for t in np.arange(T) for arm in self.arms])
        else:
            return self.R([self.beta**t*arm.compute_r(t) for arm in self.arms])
    
    def compute_believed_reward(self, t: int = None):
        '''
        Compute the reward in expectation (using rho)
        
        :param t: timestep, defaults to simulation horizon (T) if None
        :type t: int, optional
        :return: global reward (using rho)
        :rtype: float

        '''
        if t is None:
            T = len(self.arms[0].belief)
            if self.verbose:
                print(f'Computing believed reward over {T} timesteps.')
            return self.R([self.beta**t*arm.compute_rho(t) for t in np.arange(T) for arm in self.arms])
        else:
            return self.R([self.beta**t*arm.compute_rho(t) for arm in self.arms])

if __name__ == "__main__":
    pass
    
