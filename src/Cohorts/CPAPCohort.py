'''CPAPCohort: Cohort generated using CPAP data from Kang et. al. 2016, 2013 
'''
from typing import Callable, Tuple
import logging
import numpy as np
from scipy.stats import truncnorm

import src.utils as simutils
from src.Cohorts.Cohort import Cohort

class CPAPCohort(Cohort):
    '''CPAPCohort: Cohort generated using CPAP data from Kang et. al. 2016, 2013 
    '''
    def __init__(self, 
                 seed: int,
                 arm_type: str,
                 n_arms: int,
                 n_general: int,
                 n_male: int,
                 n_female: int,
                 n_adhering: int,
                 n_nonadhering: int,
                 basis_dir: str,
                 intervention_effect: Tuple[float],
                 initial_state: int = 1,
                 n_states: int = 2,
                 n_actions: int = 2, 
                 local_reward: str = 'belief_identity', 
                 lambd: float = 20.0,
                 rho: Callable = None,
                 r: Callable = None,
                 R: Callable = np.sum,
                 beta: float = 1.0,
                 sigma: float = 0.1,
                 truncate_adhering: bool = False,
                 truncate_nonadhering: bool = False,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs): 
        '''
        
        :param seed: int, seeds self.rng, an np.random.Generator used in the creation of transition matrices and seeding Arms.
        :param arm_type: str of an Arm class
        :param n_arms: int, number of Arms
        :param n_general: int, number of 'general' arms
        :param n_male: int, number of male arms
        :param n_female: int, number of female arms
        :param n_adhering: int, number of `adhering` arms according to the EM clustering in Kang et. al. 2013
        :param n_nonadhering: int, number of `non-adhering` arms according to the EM clustering in Kang et. al. 2013
        :param basis_dir: str directory to the basis transition matrices
        :param intervention_effect: (float,) effect of action!=0 on transitions (one per state). For structural constraints to be valid, must be >= 1.0
        :param initial_state: int, -1 indicates random initial state of an Arm, defaults to 1
        :param n_states: int, number of states, defaults to 2
        :param n_actions: int, number of actions, defaults to 2
        :param local_reward: str descriptor of local reward function, defaults to 'belief_identity'.
        :param lambd: float parameter of local reward, defaults to 20.0. Currently only used in local_reward = 'mate21_concave'
        :param rho: Callable, oprional rho() function
        :param r: Callable, optional local reward r(s) function
        :param R: Callable, optional global reward function R(s), defaults to np.sum
        :param beta: float discount parameter, defaults to 1.0
        :param sigma: float, standard deviation of gaussian noise added to each transition matrix.
        :param truncate_adhering: bool, if True the noise applied to any adhering arms will only shift them towards more adherence, default False
        :param truncate_nonadhering: bool, if True the noise applied to any non-adhering arms will only shift them towards less adherence, default False
        :param error_log: logging.Logger for errors.
        :param verbose: bool flag for optional console args, defaults to False
        :return:

        '''
        Cohort.__init__(self, seed=seed, arm_type=arm_type, n_arms=n_arms,
                        initial_state=initial_state, n_states=n_states, 
                        n_actions=n_actions,
                        local_reward=local_reward, lambd=lambd, rho=rho, r=r, R=R, beta=beta, 
                        verbose=verbose, **kwargs)
        
        self.n_general = n_general
        self.n_male = n_male
        self.n_female = n_female
        self.n_adhering = n_adhering
        self.n_nonadhering = n_nonadhering
        assert(self.n_arms == self.n_general + self.n_male + self.n_female + self.n_adhering + self.n_nonadhering)
        
        self.key, [self.ids_general, self.ids_male, self.ids_female, self.ids_adhering, self.ids_nonadhering] = self.assign_key(self.n_general,
                                                                                                                                self.n_male,
                                                                                                                                self.n_female,
                                                                                                                                self.n_adhering,
                                                                                                                                self.n_nonadhering,
                                                                                                                                key_label = ['general', 'male', 'female', 'adhering', 'nonadhering'])
        self.truncate_adhering = truncate_adhering
        self.truncate_nonadhering = truncate_nonadhering
        self.basis_dir = basis_dir # for loading basis matrices
        
        # Create transition matrices
        if isinstance(intervention_effect, float):
            intervention_effect = (intervention_effect,)
        if len(intervention_effect) > self.n_states:
            raise ValueError('More intervention effects are given than states in the system.')
        elif len(intervention_effect) == 1: # one intervention effect is given, so it applies to all states
            self.intervention_effect = self.n_states*intervention_effect
        else:
            self.intervention_effect = intervention_effect
        
        self.sigma = sigma
        self.transitions = self.generate_transition_matrices() # shape: NxAxSxS
        assert(self.validate_transition_matrices()) 
        
        self.arms = self.instantiate_arms()
        
    def generate_transition_matrices(self):
        '''
        Generate transition matrices that are:
            (a) valid transition matrices (see the methods of Cohort)
            (b) obey structural constraints
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix

        '''
        transitions = np.zeros((self.n_arms, self.n_actions, self.n_states, self.n_states))
        for val in set(self.key):
            self.reset_rng() # for consistent pseudorandom seed outcomes across varying attribues (and fixed seed)
            method_ = getattr(CPAPCohort, f'generate_{val}_transition_matrices')
            n_arms = getattr(self, f'n_{val}')
            mask = [True if self.key[i]==val else False for i in range(self.n_arms)]
            transitions[mask, ...] = method_(self, n_arms=n_arms)
        self.reset_rng() # reset the rng so it is consistent for other pseudorandom Cohort tasks
        return transitions
    
    def generate_transition_matrices_from_base(self, 
                                               base_arr: np.array, 
                                               n_arms: int = None, 
                                               truncate: str = None):
        '''
        Generate many transition matrices from one base_arr
        
        :param base_arr: one transition matrix
        :type base_arr: np.array
        :param n_arms: number of transition matrices to generate, defaults to None
        :type n_arms: int, optional
        :param truncate: how to truncate the normal distribution, defaults to None
        :type truncate: str, optional
        :raises ValueError: truncate must be 'left', 'right' or None
        :raises RuntimeError: invalid transitions from the combination of base_arr and noise
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix

        '''
        if n_arms is None:
            n_arms = self.n_arms 
           
        transitions = np.zeros((n_arms, self.n_actions, self.n_states, self.n_states))
        for i in range(n_arms):
            valid = False
            iters = 0
            thresh = 100
            while not valid and iters < thresh:

                if truncate == 'left':
                    # Truncates the LHS of the normal distribution, equivalent to shifting only up
                    noise = truncnorm.rvs(0, 
                                          np.inf, 
                                          loc=0.0, 
                                          scale=self.sigma, 
                                          size=(self.n_actions, self.n_states), 
                                          random_state=self.rng.integers(low=0, high=1e9))
                elif truncate == 'right':
                    # Truncates the RHS of the nomal distribution, equivalent to shifting only down
                    noise = truncnorm.rvs(-np.inf, 
                                          0, 
                                          loc=0.0, 
                                          scale=self.sigma, 
                                          size=(self.n_actions, self.n_states), 
                                          random_state=self.rng.integers(low=0, high=1e9))
                elif truncate is None:
                    noise = self.rng.normal(loc=0.0, 
                                            scale=self.sigma, 
                                            size=(self.n_actions, self.n_states))
                else:
                    raise ValueError(f'kwarg truncate={truncate} invalid')
                    
                logit = np.log(base_arr[:,:,1]/(1-base_arr[:,:,1])) + noise
                transitions[i,:,:,1] = 1/(1+np.exp(-logit))
                transitions[i,:,:,0] = 1.0 - transitions[i,:,:,1]
                valid = self.validate_structural_constraints(transitions=transitions[i,:,:,:])
                iters += 1
            if iters == thresh:
                raise RuntimeError(f'Too many iterations ({iters}) searched to find a valid transition matrix for arm at position {i}. Consider reducing noise param sigma={self.sigma}')
        
        return transitions
      
    def generate_general_transition_matrices(self, n_arms:int=None):
        '''
        Generate a transition matrix from the 'general' category
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
    
        '''
        if n_arms is None:
            n_arms = self.n_arms

        filepath = simutils.generate_filename(save_dir=self.basis_dir, 
                                              save_keyword='transitions_CPAP', 
                                              save_str='general_noAct', 
                                              save_type='npy')
        basis_arr = np.load(filepath)
        base_transition_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        base_transition_matrix[0,:,1] = basis_arr[:,1]
        base_transition_matrix[1,:,1] = self.intervention_effect*basis_arr[:,1]
        base_transition_matrix[:,:,0] = 1.0 - base_transition_matrix[:,:,1]
        
        return self.generate_transition_matrices_from_base(base_arr=base_transition_matrix, 
                                                           n_arms=n_arms)
        
    def generate_male_transition_matrices(self, n_arms:int=None):
        '''
        Generate a transition matrix from the 'male' category
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
    
        '''
        if n_arms is None:
            n_arms = self.n_arms

        filepath = simutils.generate_filename(save_dir=self.basis_dir, 
                                              save_keyword='transitions_CPAP', 
                                              save_str='male_noAct', 
                                              save_type='npy')
        basis_arr = np.load(filepath)
        base_transition_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        base_transition_matrix[0,:,1] = basis_arr[:,1]
        base_transition_matrix[1,:,1] = self.intervention_effect*basis_arr[:,1]
        base_transition_matrix[:,:,0] = 1.0 - base_transition_matrix[:,:,1]
        
        return self.generate_transition_matrices_from_base(base_arr=base_transition_matrix, 
                                                           n_arms=n_arms)

    def generate_female_transition_matrices(self, n_arms:int=None):
        '''
        Generate a transition matrix from the 'female' category
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
    
        '''
        if n_arms is None:
            n_arms = self.n_arms

        filepath = simutils.generate_filename(save_dir=self.basis_dir, 
                                              save_keyword='transitions_CPAP', 
                                              save_str='female_noAct', 
                                              save_type='npy')
        basis_arr = np.load(filepath)
        base_transition_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        base_transition_matrix[0,:,1] = basis_arr[:,1]
        base_transition_matrix[1,:,1] = self.intervention_effect*basis_arr[:,1]
        base_transition_matrix[:,:,0] = 1.0 - base_transition_matrix[:,:,1]
        
        return self.generate_transition_matrices_from_base(base_arr=base_transition_matrix, 
                                                           n_arms=n_arms)
    
    def generate_adhering_transition_matrices(self, n_arms:int=None):
        '''
        Generate a transition matrix from the 'adhering' cluster (Kang et. al. 2013)
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
    
        '''
        if n_arms is None:
            n_arms = self.n_arms

        filepath = simutils.generate_filename(save_dir=self.basis_dir, 
                                              save_keyword='transitions_CPAP', 
                                              save_str='adhering_noAct', 
                                              save_type='npy')
        basis_arr = np.load(filepath)
        base_transition_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        base_transition_matrix[0,:,1] = basis_arr[:,1]
        base_transition_matrix[1,:,1] = self.intervention_effect*basis_arr[:,1]
        base_transition_matrix[:,:,0] = 1.0 - base_transition_matrix[:,:,1]
        
        if self.truncate_adhering:
            return self.generate_transition_matrices_from_base(base_arr=base_transition_matrix, 
                                                               n_arms=n_arms, 
                                                               truncate='left')
        else:
            return self.generate_transition_matrices_from_base(base_arr=base_transition_matrix, 
                                                               n_arms=n_arms)
    
    def generate_nonadhering_transition_matrices(self, n_arms:int=None):
        '''
        Generate a transition matrix from the 'non-adhering' cluster (Kang et. al. 2013)
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
    
        '''
        if n_arms is None:
            n_arms = self.n_arms
        
        filepath = simutils.generate_filename(save_dir=self.basis_dir, 
                                              save_keyword='transitions_CPAP', 
                                              save_str='nonadhering_noAct', 
                                              save_type='npy')
        basis_arr = np.load(filepath)
        base_transition_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        base_transition_matrix[0,:,1] = basis_arr[:,1]
        base_transition_matrix[1,:,1] = self.intervention_effect*basis_arr[:,1]
        base_transition_matrix[:,:,0] = 1.0 - base_transition_matrix[:,:,1]
        
        if self.truncate_nonadhering:
            return self.generate_transition_matrices_from_base(base_arr=base_transition_matrix, 
                                                               n_arms=n_arms, 
                                                               truncate='right')
        else:
            return self.generate_transition_matrices_from_base(base_arr=base_transition_matrix, 
                                                               n_arms=n_arms)
    
    def validate_transition_matrices(self):
        '''
        Validations performed: dimensions, probabilities, and structural constraints
        
        :return: bool True if all assertions evaluate to True.
        '''
        
        assert(self.validate_dimensions())
        assert(self.validate_probabilities())
        assert(self.validate_structural_constraints())
        
        return True

if __name__ == "__main__":
    pass
    
