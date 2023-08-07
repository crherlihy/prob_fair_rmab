'''RoundRobinPolicy: pick k arms in round-robin order
'''
import logging
from src.Policies.Policy import Policy
from src.Arms.Arm import Arm

class RoundRobinPolicy(Policy):
    '''RoundRobinPolicy: pick k arms in round-robin order
    '''
    
    def __init__(self, 
                 arms: [Arm],
                 k: int,
                 horizon: int = None,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs):
        '''
        
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
        :param **kwargs: unused kwargs
        :return: None

        '''
        Policy.__init__(self, 
                        arms=arms, 
                        k=k, 
                        horizon=horizon,
                        error_log=error_log,
                        verbose=verbose, 
                        **kwargs)
       

    def select_k_arms(self, t: int, arms: [Arm] = None, k: int = None, **kwargs):
        '''Select k arms to pull in round-robin order.
        
        :param t: current timestep of simulation, int
        :param arms: arms in the simulation, defaults to self.arms
        :param k: budget of pulls. RoundRobinPolicy always pulls k arms. Defaults to self.k
        :return: list of arm indices to pull
        
        Assumes the same set of arms will be passed in, in the same order, for all timesteps t.
        '''
        if arms is None:
            arms = self.arms
        if k is None:
            k = self.k
            
        n = len(arms)
        
        return [arms[(t*k+i)%n].id for i in range(k)]


if __name__ == "__main__":
    pass
        
    