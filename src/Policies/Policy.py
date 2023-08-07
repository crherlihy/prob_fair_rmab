'''Base class for policies
'''
import logging

from src.Arms.Arm import Arm

class Policy(object):
    '''Base class for policies
    '''
    def __init__(self, 
                 horizon: int,
                 arms: [Arm],
                 k: int,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs):
        '''
        
        :param horizon: simulation horizon
        :type horizon: int
        :param arms: group of arms
        :type arms: [Arm]
        :param k: budget
        :type k: int
        :param error_log: error logger, defaults to logging.getLogger('error_log')
        :type error_log: logging.Logger, optional
        :param verbose: whether to print to the console, defaults to False
        :type verbose: bool, optional
        :param **kwargs: optional unused kwargs
        :return: None

        '''
        self.horizon = horizon
        self.arms = arms
        self.k = k
        self.error_log = error_log
        self.verbose = verbose

    def select_k_arms(self, t: int, arms: [Arm] = None, k: int = None, **kwargs):
        raise NotImplementedError("This method select_k_arms() " \
                                  "has to be implemented in the child class " \
                                  "inheriting from Policy.")
            
    def lookup_position(self, arms: [Arm] = None, arm_ids: [int] = None):
        '''
        Look up the positions of arms in self.arms, for indexing into a 
            lookup matrix, e.g. WhittleIndexPolicy.whittle_index_matrix
        :param arms: [Arm] XOR :param arm_ids: [int]
        :return: indices of arms in self.arms

        '''
        if arms:
            ids_to_lookup = [arm.id for arm in arms]
        else:
            ids_to_lookup = arm_ids
        positions = [i for i, arm in enumerate(self.arms) if arm.id in ids_to_lookup]
        return positions
        
    
if __name__ == "__main__":
    pass
