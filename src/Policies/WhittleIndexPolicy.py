'''WhittleIndexPolicy: pick k arms according to Threshold Whittle (Mate et. al.)
'''
import logging
import numpy as np

from src.Policies.Policy import Policy
from src.Arms.RestlessArm import RestlessArm

class WhittleIndexPolicy(Policy):
    '''WhittleIndexPolicy: pick k arms according to Threshold Whittle (Mate et. al.)
    '''
    def __init__(self,
                 horizon: int,
                 arms: [RestlessArm],
                 k: int,
                 arm_type: str,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False, 
                 **kwargs):
        '''
        
        :param horizon: simulation horizon
        :type horizon: int
        :param arms: group of arms
        :type arms: [RestlessArm]
        :param k: budget
        :type k: int
        :param arm_type: class of Arm, e.g. RestlessArm
        :type arm_type: str
        :param error_log: error logger, defaults to logging.getLogger('error_log')
        :type error_log: logging.Logger, optional
        :param verbose: whether to print to the console, defaults to False
        :type verbose: bool, optional
        :param **kwargs: unused kwargs
        :raises ValueError: if arm_type is an unimplemented type of Arm
        :return: None

        '''
        Policy.__init__(self, 
                        horizon=horizon, 
                        arms=arms, 
                        k=k, 
                        error_log=error_log,
                        verbose=verbose, 
                        **kwargs)
        
        if arm_type in ['RestlessArm', 'CollapsingArm']:
            self.arm_type = arm_type
        else:
            raise ValueError(f'WhittleIndexPolicy cannot handle arm_type {arm_type}.')
        
        self.whittle_index_matrix = self.compute_index(self.arms, self.horizon)
        

    @staticmethod
    def compute_belief_chains(arm: RestlessArm, horizon: int):
        """
        Compute belief chains: one chain per state in the arm's transition matrix.
        :param arm: CollapsingArm (partial observability necessitates use of belief states)
        :param horizon: total number of time steps (note that we need chains of length horizon + 1
        :return: None (belief_chains is an arm attribute and is modified in-place)
        """
        arm.belief_chains = np.zeros([arm.n_states, horizon+1])

        # At initialization, we assume that the state is s_i, where i in {0,1}.
        # For numerical reasons, we put this step in the sequence, but start processing belief chains at step t = 1
        # This is because the threshold method runs into 0**-1 error if we process the 0th step of the s0 belief chain
        for i in range(0, arm.n_states):
            arm.belief_chains[i,0] = i

        for t in range(1, horizon+1):
            for i in range(arm.n_states):
                # belief is the belief that the arm's latent state is 1 at time t
                # We pull arm at t0 (that's how we observe that belief = s_i)
                # Thus, we need to use active trans prob at t1, and passive thereafter 
                # (hence, the action_type bool)
                # belief at t_1 = belief_{t-1}*p_11a + (1-belief_{t-1})*p_01a
                # belief at t_{1<i<T} = belief_{t-1}*p_11p + (1-belief_{t-1})*p_01p
                action_type = int(t < 2)
                arm.belief_chains[i, t] = (arm.belief_chains[i, t-1]*arm.transition[action_type, 1, 1] +
                                           (1-arm.belief_chains[i, t-1])*arm.transition[action_type, 0, 1])

        return

    @staticmethod
    def compute_action_indices(arm: RestlessArm, belief_threshold: float = 0.5):
        """
        Compute x0 and x1 from the Mate et al. paper.
        (X_{omega}^{b_th} = the first belief state in each chain that is <= some belief threshold, b_th (b_th in [0,1]). 

        :param arm: arm
        :param belief_threshold: float in [0,1]
        :return: a list of indices (one per belief chain)
        
        """
        return np.ravel([np.argwhere(arm.belief_chains[i, :] <= belief_threshold)[0] for i in range(0, arm.n_states)])

    @staticmethod
    def compute_m_b(arm: RestlessArm, x0: int, x1: int):
        """
        Compute m and b from equation 6 of the Mate et al. paper.
        :param arm: arm
        :param x0: Counter for # of days since we pulled the arm and observed it to be in state 0. Used to index into belief chain 0.
        :param x1: Counter for # of days since we pulled the arm and observed it to be in state 1. Used to index into belief chain 1.
        :return: m, b
        """
        alpha = (((x1*arm.belief_chains[0, x0]) / (1 - arm.belief_chains[1, x1])) + x0)**-1
        beta = (alpha*arm.belief_chains[0, x0]) / (1 - arm.belief_chains[1, x1])
        m = alpha + beta

        rho_0 = [arm.rho(b) for b in arm.belief_chains[0, :x0]]
        rho_1 = [arm.rho(b) for b in arm.belief_chains[1, :x1]]

        b = alpha*(np.sum(rho_0)) + beta*(np.sum(rho_1)) - 1
        return m, b

    def compute_threshold(self, arm: RestlessArm, indices: [int], direction: int):
        """
        Compute either m0 or m1 from Mate et al. Algorithm 1 (which one is computed depends on the direction parameter
        :param arm: arm
        :param indices: An array of integer indices (one per state); these are the counters (time since pull and see 0, 1, ... |S|)
        :param direction: An integer indicating which counter (in the indices array) we seek to increment by 1
        :return: the threshold (subsidy) such that J_{m}^{indices} = J_{m}^{direction-modified indices'}
        """
        act_idx_new = [x+(1*(i == direction)) for i, x in enumerate(indices)]
        m0, b0 = self.compute_m_b(arm, indices[0], indices[1])
        m1, b1 = self.compute_m_b(arm, act_idx_new[0], act_idx_new[1])
        threshold = (b0 - b1)/(m0 - m1)
        return threshold

    def compute_whittle_index_one_arm(self, arm: RestlessArm, horizon: int):
        """
        Compute the Whittle index for all belief states in {0,1, ... # states} and time steps [0,1,... horizon)

        Let index be a vector
        Let time_horizon be a vector
        While there exists a state with index < time_horizon:
            Thresh = [compute_thresh() if index < time_horizon, else np.infty for state in states] #dimension 1x2
            i = argmin(thresh) # what state has min thresh?
            index_i = index[i] #how many days since pull for that state?
            w[i, index_i] = thresh[i] # accept this answer
            index[i] = index[i] + 1 #iterate in that direction
        :param arm: arm
        :param horizon: total number of time steps
        :return: Whittle index matrix (computed for the arm for all belief states and time steps)
        """

        # Initialize Whittle index matrix of dimension n_states * horizon
        whittle_index_matrix = np.zeros([arm.n_states, horizon])

        # Initialize the counters. Each counter corresponds to a state s_i and represents the # of time steps since we pulled the arm and observed it to be in state s_i
        # Note that we initialize this vector to contain 1s instead of 0s because element 0 of belief_chain[0] = 0. This causes div by 0 errors/ NaNs in subsequent calculations
        counters = np.ones([arm.n_states])

        # Precompute the change in belief for each state in transition matrix (belief_chains is an in-place arm attr)
        self.compute_belief_chains(arm, horizon)
        
        # We can't observe more than |horizon| days since pulling an arm in either state, so these counters control the while loop
        while any(x < horizon for x in counters):

            # Initialize thresholds array of dimension n_states * 1
            thresholds = np.zeros([arm.n_states])

            # Loop over states s_i; compute m_i, m_{i+1}, etc. from Mate et al. Algorithm 1
            # If a particular state's counter > horizon, put infinity for the corresponding threshold value
            for state in range(0,  arm.n_states):

                thresholds[state] = self.compute_threshold(arm,
                                                           [int(x) for i,x in enumerate(counters)],
                                                           state) if counters[state] < horizon else np.infty

            # argmin thresholds to find state (synonymous here with index (literal, not Whittle) with the min threshold
            i = np.argmin(thresholds)

            # Use the counter to look up how many days since we pulled for that state?
            index_i = int(counters[i])
            whittle_index_matrix[i, index_i] = np.min(thresholds)
            # Iterate in the  chosen direction
            counters[i] += 1

        return whittle_index_matrix

    def compute_index(self, arms: [RestlessArm], horizon: int):
        """
        Compute the Whittle index for all arms passed in from a given simulation
        :param arms: list of CollapsingArms
        :param horizon: total number of time steps
        :return: Whittle index matrix of dimension (n_arms * n_states * horizon)
        """
        whittle_index_matrix = np.zeros([len(arms), arms[0].n_states, horizon])

        for i, arm in enumerate(arms):
            whittle_index_matrix[i, :, :] = self.compute_whittle_index_one_arm(arm, horizon)
        return whittle_index_matrix

    def select_k_arms(self, t: int, arms: [RestlessArm] = None, k: int = None, **kwargs):
        """
        Largest, at least in collapsing bandits. So for example in the simulation, at time t:
        First, look up each patient's index using w[patientID, last_known_state, time_since_observed]
        Take the indices of the largest k entries in this vector, and pull those patients.
        :param t: timestep in [0, horizon)
        :param arms: set of arms to chose from, defaults to self.arms
        :param k: integral number of arms we are allowed to select, defaults to self.k
        :param arm_type: Class of Arm
        :param kwargs:
        :return: the set of top-k arm ids
        """
        if arms is None:
            arms = self.arms
        if k is None:
            k = self.k
        
        # We need a sequential set of integer ids to index into the Whittle matrix; arm id is preserved in return set
        arm_idx = self.lookup_position(arms)

        # For each arm, find the most recent timestep t where that arm was pulled
        # Then, get the state we observed via the pull by indexing into belief vector (*not belief chain*) at that t
        # Note that last_known_state is usually initialized to 1 at t=0.
        last_known_state = [arm.last_known_state for arm in arms]

        # See how many timesteps have passed since we last observed (pulled) each arm
        #time_since_observed = [t-int(np.argwhere(np.array(arm.actions) == 1)[-1]) for arm in arms]
        if self.arm_type == 'RestlessArm':
            time_since_observed = [1 for arm in arms]
        elif self.arm_type == 'CollapsingArm':
            time_since_observed = [arm.time_since_pulled for arm in arms]
        else:
            raise ValueError(f'Unknown arm_type {self.arm_type}')

        arm_info = list(zip(arm_idx, last_known_state, time_since_observed))
       
        # whittle_indices = [self.whittle_index_matrix[x[0], x[1], x[2]] for x in arm_info]
        whittle_indices = []
        for x in arm_info:
            if x[2] < self.whittle_index_matrix.shape[2]:
                whittle_indices.append(self.whittle_index_matrix[x[0], x[1], x[2]])
            else:
                # time_since_observed or time_since_pulled has reached the horizon.
                # This only occurs at the last timestep; no actions are taken.
                # We should not pick this arm again:
                whittle_indices.append(-1)

        # Get the top k arms with largest Whittle index values given last observed state and time since obs. (rel. to t)
        top_k_arm_idx = np.argsort(whittle_indices)[::-1][:k]
        top_k_arm_ids = [a.id for a in np.array(arms)[top_k_arm_idx]]  # then, get the id of each arm in top-k subset

        return top_k_arm_ids
    

if __name__ == "__main__":
    pass
