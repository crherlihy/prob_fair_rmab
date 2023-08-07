'''RandomPolicy: Randomly picks k arms at each iteration according to a uniform distribution
'''
import logging
import numpy as np

from src.Policies.Policy import Policy
from src.Arms.Arm import Arm

class RandomPolicy(Policy):
    '''RandomPolicy: Randomly picks k arms at each iteration according to a uniform distribution
    '''
    
    def __init__(self, 
                 policy_seed: int,
                 arms: [Arm],
                 k: int,
                 horizon: int = None,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs):
        '''
        
        :param policy_seed: seed of the randomized policy
        :type policy_seed: int
        :param arms: group of arms
        :type arms: [Arm]
        :param k: budget
        :type k: int
        :param horizon: simulation horizon, defaults to None
        :type horizon: int, optional
        :param error_log: error logger, defaults to logging.getLogger('error_log')
        :type error_log: logging.Logger, optional
        :param verbose: whether to print to the console, defaults to False
        :type verbose: bool, optional
        :param **kwargs: optional unused kwargs
        :return: None
        
        '''
        Policy.__init__(self, 
                        arms=arms, 
                        k=k, 
                        horizon=horizon, 
                        error_log=error_log,
                        verbose=verbose, 
                        **kwargs)
        
        self.seed = policy_seed # purely for saving in the db
        self.rng = np.random.default_rng(policy_seed)

    def select_k_arms(self, arms: [Arm] = None, k: int = None, **kwargs):
        '''Select k arms to pull.
        
        :param arms: arms in the simulation, defaults to self.arms
        :param k: budget of pulls. RandomPolicy always pulls k arms. Defaults to self.k
        :return: list of arm indices to pull
        '''
        if arms is None:
            arms = self.arms
        if k is None:
            k = self.k
        
        return [arm.id for arm in self.rng.choice(arms, k)]


if __name__ == "__main__":
    pass