'''NoActPolicy: Never acts on arms
'''
import logging

from src.Arms.Arm import Arm
from src.Policies.Policy import Policy

class NoActPolicy(Policy):
    '''NoActPolicy: Never acts on arms
    '''
    
    def __init__(self, 
                 horizon: int = None,
                 arms: [Arm] = None,
                 k: int = None,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs):
        '''
        
        :param horizon: unused kwarg of base class Policy, defaults to None
        :type horizon: int, optional
        :param arms: unused kwarg of base class Policy, defaults to None
        :type arms: [Arm], optional
        :param k: unused kwarg of base class Policy, defaults to None
        :type k: int, optional
        :param error_log: error logger, defaults to logging.getLogger('error_log')
        :type error_log: logging.Logger, optional
        :param verbose: whether to print to the console, defaults to False
        :type verbose: bool, optional
        :param **kwargs: optional unused kwargs
        :return: None

        '''
        Policy.__init__(self, 
                        horizon=horizon,
                        arms=arms,
                        k=k,
                        error_log=error_log,
                        verbose=verbose,
                        **kwargs)
        

    def select_k_arms(self, **kwargs):
        '''Select 0 arms to pull.
        
        :return: list of arm indices to pull (empty list)
        '''
        return []


if __name__ == "__main__":
    pass
