'''Arm: Base class for arms
    '''
import logging

class Arm(object):
    '''Arm: Base class for arms
    '''

    def __init__(self, 
                 id: int, 
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False, 
                 **kwargs):
        '''
        
        :param id: identifier of the arm
        :type id: int
        :param error_log: connection to error logger, defaults to logging.getLogger('error_log')
        :type error_log: logging.Logger, optional
        :param verbose: whether to print to the console, defaults to False
        :type verbose: bool, optional
        :param **kwargs: unused kwargs
        :return: None

        '''
        self.id = id
        self.error_log = error_log
        self.verbose = verbose


if __name__ == "__main__":
    pass
