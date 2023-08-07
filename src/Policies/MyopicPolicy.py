'''MyopicPolicy: pick k arms greedily,
    according to the belief of transitioning to a positive state.
'''
import logging
import numpy as np

from src.Policies.Policy import Policy
from src.Arms.RestlessArm import RestlessArm

class MyopicPolicy(Policy):
    '''MyopicPolicy: pick k arms greedily,
        according to the belief of transitioning to a positive state.
    '''
    
    def __init__(self, 
                 arms: [RestlessArm],
                 k: int,
                 horizon: int = None,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs):
        '''
        
        :param arms: group of arms
        :type arms: [RestlessArm]
        :param k: budget
        :type k: int
        :param horizon: unused required kwarg of base class Policy, defaults to None
        :type horizon: int, optional                
        :param error_log: error logger, defaults to logging.getLogger('error_log')         
        :type error_log: logging.Logger, optional
        :param verbose: whether to print to the console, defaults to False
        :type verbose: bool, optional
        :param **kwargs: unused kwargs
        :return: None

        '''
        Policy.__init__(self,  
                        horizon=horizon,
                        arms=arms,
                        k=k,
                        verbose=verbose,
                        error_log=error_log,
                        **kwargs)

    def select_k_arms(self, 
                      arms: [RestlessArm] = None, 
                      k: int = None, 
                      **kwargs):
        '''Select k arms to pull greedily.
        
        :param arms: arms in the simulation, list of class RestlessArm. Defaults to self.arms
        :param k: int budget of pulls. MyopicPolicy always pulls k arms. Defaults to self.k
        :return: list of arm.ids to pull
        '''
        if arms is None:
            arms = self.arms
        if k is None:
            k = self.k
        
        belief_if_act = np.array([arm.compute_next_belief(action=1) for arm in arms])
        belief_if_static = np.array([arm.compute_next_belief(action=0) for arm in arms])
        
        indices_to_pull = np.argsort(belief_if_act-belief_if_static)[::-1][:k]
        
        return [arms[i].id for i in indices_to_pull]

if __name__ == "__main__":
    pass
