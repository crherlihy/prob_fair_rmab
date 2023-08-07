"""IntervalHeuristicSimulation: implements periodic heuristics that ensures:
    every arm is pulled at least once in each interval of length interval_len
"""
import time
import logging
from collections import OrderedDict, defaultdict
import numpy as np

from src.Simulations.Simulation import Simulation
from src.Cohorts.Cohort import Cohort
from src.Policies.Policy import Policy


class IntervalHeuristicSimulation(Simulation):
    """IntervalHeuristicSimulation: implements periodic heuristics that ensures:
        every arm is pulled at least once in each interval of length interval_len
    """
    def __init__(self, 
                 seed: int,
                 cohort: Cohort,
                 policy: Policy, 
                 heuristic: str,
                 interval_len: int,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False, 
                 **kwargs):
        '''
        
        :param seed: int used to seed self.rng for (1) seeding arm transitions (2) the 'random' heuristic
        :param cohort: Cohort of arms to simulate
        :param policy: Policy to simulate
        :param heuristic: str descriptor of the heuristic, first/last/random
        :param interval_len: int length of an interval, must be in [1, horizon]
        :param error_log: logging.Logger for errors
        :param verbose: bool flag for optional console args, defaults to False
        :param **kwargs: 
        :return: 
            
        '''
        
        Simulation.__init__(self, 
                            seed=seed, 
                            policy=policy, 
                            cohort=cohort, 
                            error_log=error_log, 
                            verbose=verbose,
                            **kwargs)
        self.rng = np.random.default_rng(seed=seed) # if heuristic='random', used to randomly allocate pulls in intervals.
        if heuristic in ['first', 'last', 'random']:
            self.heuristic = heuristic
        else:
            raise ValueError(f'heuristic {heuristic} is not understood.')
        self.interval_len = interval_len 

    def compute_interval_lookup(self):
        """
        Helper function that computes ceil(horizon/interval_length) intervals, and then creates a look-up dictionary
        with integer timesteps in [0,horizon] as keys and the interval associated with each timestep as values.
        Example: for horizon = 10 and interval_length = 2, there are five intervals containing 2 timesteps each
        lookup[0] = 0; lookup[1] = 0; lookup[2] = 1  ... lookup[10] = 4
        :return: intervals (a range with interval_length step size), interval lookup dictionary
        """

        intervals = list(range(0, self.policy.horizon + 1, self.interval_len))
        if self.policy.horizon not in intervals:
            intervals.append(self.policy.horizon)
        interval_dict = OrderedDict()

        for i, r in enumerate(intervals[:-1]):
            for timestep in list(range(intervals[i], intervals[i + 1])):
                interval_dict[timestep] = i

        return intervals, interval_dict

    def compute_interval_pulls_remaining(self):
        """
        Helper function that computes ceil(horizon/interval_length) intervals, and then creates a look-up dictionary
        with integer timesteps in [0,horizon] as keys and the number of remaining pulls in the interval (including that
        timestep) at each timestep as values. This is a function of k as well.
        Example: for k = 3, horizon = 10 and interval_length = 2, there are four intervals containing two timesteps each.
        lookup[0] = 6; lookup[1] = 6; lookup[2] = 3  ... lookup[10] = 3
        :return: remaining interval pulls lookup dictionary
        """
        intervals = list(range(0, self.policy.horizon + 1, self.interval_len))
        if self.policy.horizon not in intervals:
            intervals.append(self.policy.horizon)

        pulls_remaining_dict = OrderedDict()
        for i, r in enumerate(intervals[:-1]):
            for timestep in list(range(intervals[i], intervals[i + 1])):
                pulls_remaining_dict[timestep] = (intervals[i + 1] - timestep) * self.policy.k

        return pulls_remaining_dict

    def compute_random_constrained_pulls(self):
        """
        Helper function for 'random' heuristic that computes ceil(horizon/interval_length) intervals, and then creates a look-up dictionary
        with integer timesteps in [0,horizon] as keys and the number of assigned constrained pulls to each timestep in
        [0,k]. The constrained pulls are distributed randomly within each interval, and independently between intervals.
        Example: for k = 4, n_arms = 5, and interval_length = 2, there are four intervals containing two timesteps each.
        lookup[0] = 4; lookup[1] = 1; lookup[2] = 3, lookup[3] = 2 ...
        Note that since a defaultdict is used, timesteps with 0 constrained pulls will correctly return 0 at runtime
        :return: constrained pulls lookup dictionary
        """
        intervals = list(range(0, self.policy.horizon + 1, self.interval_len))
        if self.policy.horizon not in intervals:
            intervals.append(self.policy.horizon)

        random_pulls_dict = defaultdict(int)
        for i, r in enumerate(intervals[:-1]):
            rands = self.rng.choice(a=self.policy.k * (intervals[i + 1] - intervals[i]), 
                                    size=self.cohort.n_arms, 
                                    replace=False)
            for rand in rands:
                curr_t = intervals[i] + int(rand / self.policy.k)
                random_pulls_dict[curr_t] += 1

        return random_pulls_dict

    def run(self, iterations: int = None, debug: bool = False):
        '''
        Run the simulation from timestep 0 to timestep self.policy.horizon-1
            Overrides base class Simulation.run()
        Constraint: ensure an arm is pulled at least once in each interval self.interval_len
        Heuristic: satisfy constriant using heuristic self.heuristic
            else, pull optimally.

        :return: tuple (actions, adherence)
            actions: numpy array of shape NxT, action for each arm at time t
            adherence: numpy array of shape NxT+1, state for each arm at time t
                adherence includes an initial state s_0
        '''
        if iterations is None:
            iterations=self.iterations
            
        actions = np.zeros((iterations, self.cohort.n_arms, self.policy.horizon), dtype='int64')
        adherence = np.zeros((iterations, self.cohort.n_arms, self.policy.horizon + 1), dtype='int64')
        runtimes = np.zeros(iterations, dtype='float64')
        
        intervals, interval_lookup = self.compute_interval_lookup()

        interval_pulls_remaining_dict = self.compute_interval_pulls_remaining()
        arm_ids_dict = {arm.id: arm for arm in self.cohort.arms}
        
        simulation_seeds = self.generate_seeds(iterations)
            # Note that if we call this function multiple times, the same seeds will be returned. This is intended behavior for reproducibility.
        
        for iteration in np.arange(iterations, dtype='int64'):
            constrained_pulls_dict = self.compute_random_constrained_pulls() # only used if the heuristic is 'random'

            self.cohort.reset_arms(seeds=simulation_seeds[iteration, :])

            periodic_constraint_sat = np.zeros([self.cohort.n_arms, len(intervals)])
            
            start_time = time.time()
        
            for t, interval_id in interval_lookup.items():
                available_arms = [a for i, a in enumerate(self.cohort.arms) if periodic_constraint_sat[i, interval_id] == 0]
                if debug:
                    print(f't={t}, interval={interval_id}')
                if self.heuristic == 'first':

                    # First check if all k pulls must be constrained
                    if len(available_arms) >= self.policy.k:
                        arm_ids = self.policy.select_k_arms(t=t, arms=available_arms, k=self.policy.k)

                    # Next check if all constrained pulls have been satisfied
                    elif len(available_arms) == 0:
                        arm_ids = self.policy.select_k_arms(t=t, arms=self.cohort.arms, k=self.policy.k)

                    # If the # of constrained arms remaining is <k, then we use all of those plus some optimal arms
                    else:
                        remaining_arms = [arm for arm in self.cohort.arms if arm not in available_arms]
                        arm_ids = [arm.id for arm in available_arms] + \
                                  self.policy.select_k_arms(t=t, 
                                                            arms=remaining_arms, 
                                                            k=self.policy.k - len(available_arms))

                elif self.heuristic == 'last':
                    optimal_pulls_remaining = interval_pulls_remaining_dict[t] - len(available_arms)

                    # First check if all pulls must be constrained
                    if optimal_pulls_remaining <= 0:
                        arm_ids = self.policy.select_k_arms(t=t, arms=available_arms, k=self.policy.k)

                    # Next check if no pulls must be constrained
                    elif optimal_pulls_remaining >= self.policy.k:
                        arm_ids = self.policy.select_k_arms(t=t, arms=self.cohort.arms, k=self.policy.k)

                    # Step through one by one, using optimal pulls until remaining pulls must be constrained
                    else:
                        timestep_pulls_remaining = self.policy.k
                        arm_ids = []
                        all_arms = [arm for arm in self.cohort.arms]
                        for i in range(0, self.policy.k):
                            if optimal_pulls_remaining > 0:
                                arm_id = self.policy.select_k_arms(t=t, arms=all_arms, k=1)[0]  # Pulling top one arm
                                selected_arm = arm_ids_dict[arm_id]
                                arm_ids.append(arm_id)
                                all_arms.remove(selected_arm)  # Can't pull same arm twice
                                if selected_arm in available_arms:  # Satisfied constraint in an optimal pull
                                    available_arms.remove(selected_arm)
                                else:
                                    optimal_pulls_remaining -= 1
                                timestep_pulls_remaining -= 1
                            else:  # No more optimal pulls remaining
                                break
                        if timestep_pulls_remaining > 0:  # Conduct remaining pulls, all of which are constrained
                            arm_ids += self.policy.select_k_arms(t=t, 
                                                                 arms=available_arms, 
                                                                 k=timestep_pulls_remaining)

                elif self.heuristic == 'random':
                    if debug:
                        print(f'available arms: {[arm.id for arm in available_arms]}')
                        print(f'Number of constrained pulls: {constrained_pulls_dict[t]}')

                    num_constrained_pulls = constrained_pulls_dict[t]
                    
                    # First check if all k pulls must be constrained and we have sufficient unsatisfied arms
                    if num_constrained_pulls == self.policy.k and len(available_arms) >= self.policy.k:
                        if debug:
                            print('All constrained pulls')
                        arm_ids = self.policy.select_k_arms(t=t, arms=available_arms, k=self.policy.k)

                    # Check if all arms have been satisfied or constraint = 0
                    elif num_constrained_pulls == 0 or len(available_arms) == 0:
                        if debug:
                            print('No constrained pulls')
                        arm_ids = self.policy.select_k_arms(t=t, arms=self.cohort.arms, k=self.policy.k)

                    # If min(len(available_arms), num_constrained_pulls) is in [1,k-1]
                    else:
                        if debug:
                            constr_pulls = min(len(available_arms), num_constrained_pulls)
                            print("Splitting pulls between constrained and unconstrained")
                            print(f"Num constrained pulls: {constr_pulls}")
                            print(f"Unconstrained: {self.policy.k - constr_pulls}")
                        num_arms_to_pull = min(len(available_arms), num_constrained_pulls)
                        arm_ids_1 = self.policy.select_k_arms(t=t, 
                                                              arms=available_arms, 
                                                              k=num_arms_to_pull)  # constrained pulls
                        remaining_arms = [arm for arm in self.cohort.arms if arm.id not in arm_ids_1]
                        arm_ids_2 = self.policy.select_k_arms(t=t, 
                                                              arms=remaining_arms, 
                                                              k=self.policy.k - num_arms_to_pull)  # unconstrained pulls
                        if debug:
                            print(f'Constrained pulls: {arm_ids_1},\n Remaining arms:{[arm.id for arm in remaining_arms]}, \n Unconstrained pulls: {arm_ids_2}')
                        arm_ids = arm_ids_1 + arm_ids_2
    
                else:  # No constraint sequence
                    # Select the top-k arms using whatever selection method our policy implements; arm_ids :: [int]
                    arm_ids = self.policy.select_k_arms(t=t, 
                                                        arms=self.cohort.arms, 
                                                        k=self.policy.k)
    
                # Save actions and adherences at time t
                # Recall action[t] goes from state[t] to state[t+1]
                adherence[iteration, :, t] = [arm.state[t] for arm in self.cohort.arms]
                actions[iteration, :, t] = [1 if arm.id in arm_ids else 0 for arm in self.cohort.arms]
    
                # Now, perform the selected action on each arm
                self.cohort.update_arms(actions=actions[iteration,:,t])
                periodic_constraint_sat[arm_ids, interval_id] = 1
                
            # save the last state
            adherence[iteration, :, self.policy.horizon] = [arm.state[self.policy.horizon] for arm in self.cohort.arms]
            runtimes[iteration] = time.time() - start_time
        return actions, adherence, runtimes

if __name__ == "__main__":
    pass
    