'''SyntheticCohort: Cohort generated from synthetic parameters
'''
from typing import Callable
import logging
import numpy as np

from src.Cohorts.Cohort import Cohort
from src.Arms.Arm import  Arm

class SyntheticCohort(Cohort):
    '''SyntheticCohort: Cohort generated from synthetic parameters
    '''
    def __init__(self, 
                 seed: int, 
                 arm_type: str,
                 n_arms: int,
                 n_forward: int,
                 n_reverse: int,
                 n_concave: int,
                 n_convex: int,
                 n_random: int = None,
                 initial_state: int = 1,
                 n_states: int = 2,
                 n_actions: int = 2, 
                 local_reward: str = 'belief_identity',
                 R: Callable = np.sum,
                 beta: float = 1.0,
                 override_threshold_optimality_conditions: bool = True,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs): 
        '''
        
        :param seed: int, seeds self.rng, an np.random.Generator used in the creation of transition matrices and seeding Arms.
        :param arm_type: str of an Arm class
        :param n_arms: int, number of Arms
        :param n_forward: int, number of forward threshold optimal arms
        :param n_reverse: int, number of reverse threshold optimal arms
        :param n_concave: int, number of concave arms (in the context of the ProbFair policy)
        :param n_convex: int, number of strictly convex arms (in the context of the ProbFair policy)
        :param n_random: int, number of randomly generated arms that obey structural constraints, but not necessarily any other property
        :param initial_state: int, -1 indicates random initial state of an Arm, defaults to 1
        :param n_states: int, number of states, defaults to 2
        :param n_actions: int, number of actions, defaults to 2
        :param local_reward: str descriptor of local reward function, defaults to 'belief_identity'. Valid values: 'belief_identity', 'mate21_concave'
        :param R: Callable, optional global reward function R(s), defaults to np.sum
        :param beta: float discount parameter, defaults to 1.0
        :param override_threshold_optimality_conditions: bool flag, defaults to True. If True, uses 'belief_identity' to determine threshold optimality
        :param error_log: logging.Logger for errors.
        :param verbose: bool flag for optional console args, defaults to False
        :param **kwargs:
            lambd is a kwarg passed to Cohort, default 20.0
            If r, rho are kwargs, they are passed to Cohort and become self.r, self.rho. 
                The user should ensure that r, rho match local_reward (required param)
        :return: None

        To allow any rho(), we will need to find min, max rho'(b) over b \in [0,1].
        Thus, we only have the following two local reward functions implemented:
            belief_identity, mate21_concave.
        
        Given the same seed, cohorts with different n_keys are comparable--transition matrix generation is determined by seed. 
        Recall forward threshold optimal implies concave and reverse threshold optimal implies convex.
        '''
        
        Cohort.__init__(self, seed=seed, arm_type=arm_type, n_arms=n_arms, 
                        initial_state=initial_state, n_states=n_states, 
                        n_actions=n_actions,
                        local_reward=local_reward, R=R, beta=beta, 
                        error_log=error_log, verbose=verbose, **kwargs)
        
        # It is conjectured that forward threshold-optimality implies concavity
        # and reverse threshold-optimality implies strict convexity
        assert(n_forward <= n_concave)
        assert(n_reverse <= n_concave)
        self.n_forward = n_forward
        self.n_reverse = n_reverse
        self.n_concave = n_concave
        self.n_convex = n_convex
        
        if n_random is not None:
            self.n_random = n_random
        else:
            self.n_random = self.n_arms - self.n_convex - self.n_concave
            if verbose:
                print(f'Setting {self.n_random} random arms.')
        assert(self.n_arms == self.n_forward+self.n_reverse+(self.n_concave-self.n_forward)+(self.n_convex-self.n_reverse)+self.n_random)
        assert(self.n_arms == self.n_concave+self.n_convex+self.n_random)
        
        if self.verbose:
            print(f'Creating a cohort with N={self.n_arms} arms, of which {self.n_concave} concave ({self.n_forward} forward), {self.n_convex} convex ({self.n_reverse} reverse), and {self.n_random} random.')
        
        self.key, \
            [self.ids_forward, 
             self.ids_reverse, 
             self.ids_concave, 
             self.ids_convex, 
             self.ids_random] = self.assign_key(self.n_forward, 
                                                self.n_reverse, 
                                                self.n_concave-self.n_forward, 
                                                self.n_convex-self.n_reverse, 
                                                self.n_random, 
                                                key_label=['forward', 'reverse', 'concave', 'convex', 'random'])

        # Create transition matrices
        self.override_threshold_optimality_conditions = override_threshold_optimality_conditions
        if self.override_threshold_optimality_conditions:
            self.threshold_optimality_evaluation_method = 'belief_identity'
        else:
            self.threshold_optimality_evaluation_method = self.local_reward
            
        self.transitions = self.generate_transition_matrices() # shape: NxAxSxS
        assert(self.validate_transition_matrices())
        
        self.arms = self.instantiate_arms()
      
        
    def generate_transition_matrices(self):
        '''
        Generate transition matrices that are:
            (a) valid transition matrices (see the methods of Cohort)
            (b) obey structural constraints
            (c) forward/reverse threshold optimal when applicable
            (d) concave/convex when applicable
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix

        '''
        transitions = np.zeros((self.n_arms, self.n_actions, self.n_states, self.n_states))
        for key_type in set(self.key):
            self.reset_rng() # for consistent pseudorandom seed outcomes across varying attribues (and fixed seed)
            method_ = getattr(SyntheticCohort, f'generate_{key_type}_transition_matrices')
            # n_arms = getattr(self, f'n_{val}')
            n_key = len([True for i in range(self.n_arms) if self.key[i]==key_type])
            mask = [True if self.key[i]==key_type else False for i in range(self.n_arms)]
            transitions[mask, ...] = method_(self, n_arms=n_key)
        self.reset_rng() # reset the rng so it is consistent for other pseudorandom Cohort tasks
        return transitions
    
    def generate_random_transition_matrices(self, n_arms: int = None):
        '''
        Generate a random transition matrix that ensures structural constraints are enforced
            Overloads Cohort's generate_random_transition_matrix())
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
    
        '''
        if n_arms is None:
            n_arms = self.n_arms
            
        assert(self.n_actions==2) # This is not implemented for n_actions != 2
        
        transitions = np.zeros((n_arms, self.n_actions, self.n_states, self.n_states))
        for i in range(n_arms):
            rand_vars = self.rng.uniform(low=0, high=1, size=self.n_actions*self.n_states)
            transitions[i,0,0,1] = np.min(rand_vars)
            transitions[i,1,1,1] = np.max(rand_vars)
            rand_vars = np.delete(rand_vars, np.argmin(rand_vars))
            rand_vars = np.delete(rand_vars, np.argmax(rand_vars))
            transitions[i,0,1,1] = rand_vars[0]
            transitions[i,1,0,1] = rand_vars[1]
            transitions[i,:,:,0] = 1 - transitions[i,:,:,1]
        
        return transitions

    def generate_forward_transition_matrices(self, n_arms: int = None):
        '''
        Generate a forward threshold optimal transition matrix
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
        
        '''
        if n_arms is None:
            n_arms = self.n_arms
            
        assert(self.n_actions==2 and self.n_states==2)
        
        transitions = np.zeros((n_arms, self.n_actions, self.n_states, self.n_states))
        for i in range(n_arms):
            valid = False
            while not valid:
                transition_candidate = self.generate_random_transition_matrices(n_arms=1)
                valid = self.validate_forward_threshold_optimal(transition_candidate)
            transitions[i,...] = transition_candidate
        return transitions
    
    def generate_reverse_transition_matrices(self, n_arms: int = None):
        '''
        Generate a reverse threshold optimal transition matrix
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix

        '''
        if n_arms is None:
            n_arms = self.n_arms
            
        assert(self.n_actions==2 and self.n_states==2)
        
        transitions = np.zeros((n_arms, self.n_actions, self.n_states, self.n_states))
        for i in range(n_arms):
            valid = False
            while not valid:
                transition_candidate = self.generate_random_transition_matrices(n_arms=1)
                valid = self.validate_reverse_threshold_optimal(transition_candidate)
            transitions[i,...] = transition_candidate
        return transitions
    
    def generate_concave_transition_matrices(self, n_arms: int = None):
        '''
        Generate a concave transition matrix
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
        '''
        if n_arms is None:
            n_arms = self.n_arms
            
        assert(self.n_actions==2 and self.n_states==2)
        
        transitions = np.zeros((n_arms, self.n_actions, self.n_states, self.n_states))
        for i in range(n_arms):
            valid = False
            while not valid:
                transition_candidate = self.generate_random_transition_matrices(n_arms=1)
                valid = self.validate_concave(transition_candidate)
            transitions[i,...] = transition_candidate
        return transitions
    
    def generate_convex_transition_matrices(self, n_arms: int = None):
        '''
        Generate a convex transition matrix
        
        :param n_arms: number of arms (int), defaults to self.n_arms
        :return: (n_arms x n_actions x n_states x n_states) numpy transition matrix
        '''

        if n_arms is None:
            n_arms = self.n_arms
            
        assert(self.n_actions==2 and self.n_states==2)
        
        transitions = np.zeros((n_arms, self.n_actions, self.n_states, self.n_states))
        for i in range(n_arms):
            valid = False
            while not valid:
                transition_candidate = self.generate_random_transition_matrices(n_arms=1)
                valid = self.validate_convex(transition_candidate)
            transitions[i,...] = transition_candidate
        return transitions
    
    
    def validate_transition_matrices(self):
        '''
        Validations performed: dimensions, probabilities, structural constraints, 
         and threshold optimality conditions (where applicable)
        
        :return: bool True if all assertions evaluate to True.
        '''

        assert(self.validate_dimensions())
        assert(self.validate_probabilities())
        assert(self.validate_structural_constraints())
        assert(self.validate_forward_threshold_optimal(self.transitions[np.in1d(self.key, ['forward'])]))
        assert(self.validate_reverse_threshold_optimal(self.transitions[np.in1d(self.key, ['reverse'])]))
        assert(self.validate_concave(self.transitions[np.in1d(self.key, ['forward', 'concave'])]))
        assert(self.validate_convex(self.transitions[np.in1d(self.key, ['reverse', 'convex'])]))
        return True
    
    def validate_forward_threshold_optimal(self, transitions: np.array = None):
        '''
        Validation of cohort: transitions are forward threshold optimal
        
        :param transitions: numpy transition matrix, default self.transitions
        :return: True if:
            rho='belief_identity'
            (a) passive_intertia_term + active_inertia_term < 1/beta
            (b) passive_inertia_term > active_inertia_term
            
            or
            
            rho='mate21_concave'
            (a) passive_inertia_term(1-beta(max))/active_inertia_term(1-beta(min)) >= e^abs(lambda)
            
        '''
        
        if np.size(transitions) == 0:
            # Then we've got an empty slice of an array
            return True

        if transitions is None:
            transitions = self.transitions
        
        shape = np.shape(transitions)
        assert(shape[-1]==2 and shape[-3]==2) # inequalities only supported for two-state and two-action system.
        if len(shape) == 3:
            transitions = np.reshape(transitions,(-1,)+shape)
        assert(self.beta>0 and self.beta<=1)
        
        passive_inertia_term =  transitions[:,0,1,1] - transitions[:,0,0,1]
        active_inertia_term =  transitions[:,1,1,1] - transitions[:,1,0,1]
        
        if self.threshold_optimality_evaluation_method=='belief_identity':
            indexability = np.all(passive_inertia_term + active_inertia_term < 1/self.beta)
            forward = np.all(passive_inertia_term > active_inertia_term)
            return indexability & forward
        
        elif self.threshold_optimality_evaluation_method=='mate21_concave':
            minimum = np.min((passive_inertia_term, active_inertia_term))
            maximum = np.max((passive_inertia_term, active_inertia_term))
            LHS = (passive_inertia_term*(1-self.beta*maximum))/(active_inertia_term*(1-self.beta*minimum))
            forward = np.all(LHS>=np.exp(np.abs(self.lambd)))
            return forward
    
        else:
            raise ValueError(f'validate_forward_thresh_opt() is not implemented for {self.threshold_optimality_evaluation_method}')
        
        return 
    
    def validate_reverse_threshold_optimal(self, transitions: np.array = None):
        '''
        Validate a transition matrix (or numpy matrix of transition matrices).
        Validation: matrix is reverse threshold optimal
        
        :param transitions: numpy transition matrix, default self.transitions
        :return: True if:
            rho='belief_identity'
            (a) passive_intertia_term + active_inertia_term < 1/beta
            (b) passive_inertia_term < active_inertia_term
            
            or
            
            rho='mate21_concave'
            (a) passive_inertia_term(1-beta(min))/active_inertia_term(1-beta(max)) <= e^abs(lambda)
    
        '''
        if np.size(transitions) == 0:
            # Then we've got an empty slice of an array
            return True
        
        if transitions is None:
            transitions = self.transitions
        
        shape = np.shape(transitions)
        assert(shape[-1]==2 and shape[-3]==2) # inequalities only supported for two-state and two-action system.
        if len(shape) == 3:
            transitions = np.reshape(transitions,(-1,)+shape)
        assert(self.beta>0 and self.beta<=1)
        
        passive_inertia_term =  transitions[:,0,1,1] - transitions[:,0,0,1]
        active_inertia_term =  transitions[:,1,1,1] - transitions[:,1,0,1]
        
        if self.threshold_optimality_evaluation_method=='belief_identity':
            indexability = np.all(passive_inertia_term + active_inertia_term < 1/self.beta)
            reverse = np.all(passive_inertia_term < active_inertia_term)
            return indexability & reverse
        
        elif self.threshold_optimality_evaluation_method=='mate21_concave':
            minimum = np.min((passive_inertia_term, active_inertia_term))
            maximum = np.max((passive_inertia_term, active_inertia_term))
            LHS = (passive_inertia_term*(1-self.beta*minimum))/(active_inertia_term*(1-self.beta*maximum))
            reverse = np.all(LHS<=np.exp(np.abs(self.lambd)))
            return reverse
    
        else:
            raise ValueError(f'validate_reverse_thresh_opt() is not implemented for {self.threshold_optimality_evaluation_method}')
        
        return 
    
    def validate_concave(self, transitions: np.array = None):
        '''
        Validate that all of the transition matrices are concave
        
        :param transitions: the transitions to test, defaults to self.transitions if None
        :type transitions: np.array, optional
        :return: True if all transitions are valid
        :rtype: bool

        '''
        
        if np.size(transitions) == 0:
            # Then we've got an empty slice of an array
            return True
        
        if transitions is None:
            transitions = self.transitions
        
        shape = np.shape(transitions)
        assert(shape[-1]==2 and shape[-3]==2) # two-state and two-action system supported (only)
        if len(shape) == 3:
            transitions = np.reshape(transitions,(-1,)+shape)
        
        return np.all([self.get_f_concavity(transitions[i,:,:,:]) == "concave" for i in range(transitions.shape[0])])
    
    def validate_convex(self, transitions: np.array = None):
        '''
        Validate that all of the transition matrices are convex
        
        :param transitions: the transitions to test, defaults to self.transitions if None
        :type transitions: np.array, optional
        :return: True if all transitions are valid
        :rtype: bool

        '''
        
        if np.size(transitions) == 0:
            # Then we've got an empty slice of an array
            return True
        if transitions is None:
            transitions = self.transitions
        
        shape = np.shape(transitions)
        assert(shape[-1]==2 and shape[-3]==2) # two-state and two-action system supported (only)
        if len(shape) == 3:
            transitions = np.reshape(transitions,(-1,)+shape)
        
        return np.all([self.get_f_concavity(transitions[i,:,:,:]) == "convex" for i in range(transitions.shape[0])])

    @staticmethod
    def get_f_concavity(transition: np.array) -> str:
        '''
        Get the concavity/linear or convexity of one arm using h(p, x0, interval_len)
        
        @param transition: Transition matrix to assess
        @return: label in {concave, convex}
        '''
        a0 = transition[0, 0, 1]  # P_{0,1}^0
        a1 = transition[1, 0, 1] - transition[0, 0, 1]  # P_{0,1}^1 - P_{0,1}^0
        b0 = 1 - transition[0, 1, 1] + transition[0, 0, 1]  # 1 - P_{1,1}^0 + P_{0,1}^0
        b1 = transition[0, 1, 1] - transition[1, 1, 1] \
             - transition[0, 0, 1] + transition[1, 0, 1]  # P_{1,1}^0 - P_{1,1}^1 - P_{0,1}^0 + P_{0,1}^1

        if b1 == 0:
            return "concave"
        elif b1 != 0 and a1 == 0:
            return "concave" if a0 == 0 else "convex"
        else:
            return "convex" if a0 > (a1 * b0) / b1 else "concave"
        
if __name__ == "__main__":
    pass
   