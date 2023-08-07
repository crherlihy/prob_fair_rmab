"""Simulation: base class for simulations.
"""
import time
import logging
import numpy as np

from src.Cohorts.Cohort import Cohort
from src.Policies.Policy import Policy
class Simulation(object):
    """Simulation: base class for simulations.
    """
    def __init__(self, 
                 seed: int,
                 cohort: Cohort,
                 policy: Policy, 
                 iterations: int = 1,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False, 
                 **kwargs):
        '''
        
        :param seed: seeds the arm transitions that occur
        :param cohort: Cohort of arms to simulate
        :param policy: Policy to simulate
        :param iterations: Number of times simulation should run, default 1
        :param error_log: logging.Logger for errors
        :param verbose: bool flag for optional console args, defaults to False
        :return:

        '''
        self.error_log = error_log
        self.seed = seed # not setting the rng here for arm consistency, see generate_seeds()
        self.cohort = cohort
        self.policy = policy
        self.iterations = iterations
        self.verbose = verbose
        self.simulation_seeds = None


    def generate_seeds(self, iterations: int = None):
        '''
        Generate rng seeds for arms
        
        :param iterations: number of simulation iterations, defaults to self.iterations
        :type iterations: int, optional
        :return: iterations x n_arms seeds to use
        :rtype: np.array

        '''
        if iterations is None:
            iterations = self.iterations
        rng = np.random.default_rng(seed=self.seed)
        return rng.integers(low=0, high=10**5, size=(iterations, self.cohort.n_arms))

    def run(self, iterations: int = None):
        '''
        Run the simulation from timestep 0 to timestep self.policy.horizon-1
        
        :return: tuple (actions, adherence)
            actions: numpy array of shape NxT, action for each arm at time t
            adherence: numpy array of shape NxT+1, state for each arm at time t 
                adherence includes an initial state s_0

        '''
        if iterations is None:
            iterations=self.iterations
            
        actions = np.zeros((iterations, self.cohort.n_arms, self.policy.horizon), 
                           dtype='int64')
        adherence = np.zeros((iterations, self.cohort.n_arms, self.policy.horizon + 1), 
                             dtype='int64')
        runtimes = np.zeros(iterations, dtype='float64')

        self.simulation_seeds = self.generate_seeds(iterations)
            # Note that if we call this function multiple times, the same seeds will be returned. This is intended behavior for reproducibility.
        
        for iteration in np.arange(iterations, dtype='int64'):
            self.cohort.reset_arms(seeds=self.simulation_seeds[iteration, :])

            start_time = time.time()
                             
            for t in range(0, self.policy.horizon):
                # Select the top-k arms using whatever selection method our policy implements; arm_ids :: [int]
                selected_arm_ids = self.policy.select_k_arms(t=t, 
                                                             arms=self.cohort.arms, 
                                                             k=self.policy.k)
    
                # Save actions and adherences at time t
                # Recall action[t] goes from state[t] to state[t+1]
                adherence[iteration, :, t] = [arm.state[t] for arm in self.cohort.arms]
                actions[iteration, :, t] = [1 if arm.id in selected_arm_ids else 0 for arm in self.cohort.arms]
    
                # Now, perform the selected action on each arm
                self.cohort.update_arms(actions=actions[iteration,:,t])
    
            # Save the state after the last pull
            adherence[iteration,:, self.policy.horizon] = [arm.state[self.policy.horizon] for arm in self.cohort.arms]

            # Save the runtime (in seconds) over the entire time horizon, T, for this iteration
            runtimes[iteration] = time.time() - start_time
            
        return actions, adherence, runtimes

if __name__ == "__main__":
    pass
