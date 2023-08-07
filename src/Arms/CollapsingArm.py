'''CollpasingArm: a type of RestlessArm where states are partially observable.
'''
import numpy as np

from src.Arms.RestlessArm import RestlessArm

class CollapsingArm(RestlessArm):
    '''CollpasingArm: a type of RestlessArm where states are partially observable.
    
    State information is known iff action==1 is taken. 
    We assume the initial state is known (even if random).
    '''
    def __init__(self, **kwargs):
        '''
        
        :param **kwargs: RestlessArm keyword args
        :return: None

        '''
        RestlessArm.__init__(self, **kwargs) 

    def _update_belief_state(self):
        '''
        Updates (appends) in place:
            self.belief with self.state
        Must be called after self._update_true_state(), see self.update()

        '''
        if len(self.belief) != len(self.state) - 1:
            raise ValueError('Ensure self._update_true_state() has been called first')
        
        action = self.actions[-1]
        if action == 1:
            # Then, the arm's state is observed and belief is updated accordingly:
            self.belief[-1] = self.state[-2]
            self.last_known_state = self.belief[-1]  # We only observe the state when we pull. This is our most recent pull; update last known state.

        next_belief = self.compute_next_belief(action)
        self.belief.append(next_belief)

if __name__ == '__main__':
    pass

