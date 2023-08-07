# -*- coding: utf-8 -*-
### Runs a simulation.
import os
import configparser
import logging
import time
from pydoc import locate
from typing import Callable
from collections import OrderedDict

import src.utils as simutils
from src.Database.write import DBWriter
import src.Database.get_foreign_key_vals as get_foreign_key_vals

def run_simulation(DB_WRITER: DBWriter,
                   SIM_ID: int,
                   COHORT_TYPE: str,
                   ARM_TYPE: str,
                   N_ARMS: int,
                   COHORT_SEED: int, 
                   LOCAL_REWARD: str,
                   GLOBAL_REWARD: Callable,
                   REWARD_DISCOUNT: float,
                   INITIAL_STATE: int,
                   POLICY_TYPE: str,
                   K: int,
                   HORIZON: int,
                   SIMULATION_TYPE: str,
                   SIMULATION_SEED: int, 
                   SIMULATION_ITERATIONS: int, 
                   LOG_SIMULATION_FLAG: bool = True,
                   error_log: logging.Logger = logging.getLogger('error_log'),
                   VERBOSE_FLAG: bool = False,
                   **kwargs):
    '''
    Run a simulation.

    :param DB_WRITER: DBWriter object; writes experiment, simulation, cohort, and policy information to mysqldb.
    :param SIM_ID: int
    :param COHORT_TYPE: str class name of a Cohort class
    :param ARM_TYPE: str class name of an Arm class
    :param N_ARMS: int number of arms
    :param COHORT_SEED: int seed of the cohort. Governts transition matrix generation; key order (e.g. the keys forward/reverse/random)
    :param LOCAL_REWARD: str describing the local reward function r(s)/rho(b)
    :param GLOBAL_REWARD: Callable global reward function R()
    :param REWARD_DISCOUNT: (beta) float discount parameter of reward
    :param INITIAL_STATE: int initial state in arange(n_states), if -1 randomly assigned.
    :param POLICY_TYPE: str class name of a Policy class
    :param K: int pull budget k
    :param HORIZON: int horizon length (T)
    :param SIMULATION_TYPE: str class name of a Simulation class
    :param SIMULATION_SEED: int seed of a simulation, governs the transitions that occur and the random heuristic of IntervalHeuristicSimulation
    :param SIMULATION_ITERATIONS: int number of times to repeat a simulation (bootstrapping)
    :param error_log: error logger, defaults to logging.getLogger('error_log')
    :param VERBOSE_FLAG: bool whether to print optional statements to the console, defaults to False
    :param **kwargs: The following optional kwargs are allowed:
        (Cohort) lambd, rho, r, n_actions, n_states, pull_action
        (CPAPCohort) n_male, n_female, basis_dir, intervention_effect, sigma
        (SyntheticCohort) n_forward, n_reverse, n_random, override_threshold_optimality_conditions
        (MathProgPolicy) interval_len, min_sel_frac, min_pull_per_pd, type, n_actions (a Cohort kwarg), pull_action (a Cohort kwarg)
        (ProbFairPolicy) prob_pull_lower_bound, prob_pull_upper_bound, ncut, pull_action (a Cohort kwarg)
        (RandomPolicy) policy_seed
        (WhittleIndexPolicy) arm_type (a Cohort kwarg)
        (IntervalHeuristicSimulation) heuristic, interval_len (a Policy kwarg)
    :return: actions, adherences

    '''
    
    ### Kwargs to lowercase so they can properly be passed in to Cohort, Policy, Simulation classes
    lowercase_keys = [k for k,v in kwargs.items() if k.lower()==k]
    if lowercase_keys:
        raise ValueError(f'The following kwarg keys need to be passed in in ALL_CAPS: {lowercase_keys}.')
    keyword_args = {k.lower(): v for k,v in kwargs.items()}
    
    ### Create cohort
    try:
        if VERBOSE_FLAG:
            print(f'Instantiating {COHORT_TYPE}.')
        class_ = locate(f'src.Cohorts.{COHORT_TYPE}.{COHORT_TYPE}')
        cohort = class_(seed=COHORT_SEED,
                        arm_type=ARM_TYPE,
                        n_arms=N_ARMS,
                        initial_state=INITIAL_STATE,
                        local_reward=LOCAL_REWARD,
                        R=GLOBAL_REWARD,
                        beta=REWARD_DISCOUNT,
                        error_log=error_log,
                        verbose=VERBOSE_FLAG,
                        **keyword_args)
    except:
        raise ValueError(f'Unable to instantiate Cohort {COHORT_TYPE}')

    ### Create policy
    try:
        if VERBOSE_FLAG:
            print(f'Instantiating {POLICY_TYPE}.')
        class_ = locate(f'src.Policies.{POLICY_TYPE}.{POLICY_TYPE}')

        policy_start_time = time.time()
        policy = class_(arms=cohort.arms, 
                        arm_type=ARM_TYPE,
                        k=K,
                        horizon=HORIZON,
                        error_log=error_log,
                        verbose=VERBOSE_FLAG,
                        **keyword_args)
        policy_init_time = time.time() - policy_start_time
    except:
        raise ValueError(f'Unable to instantiate Policy {POLICY_TYPE}')
    
    ### Create simulation
    try:
        if VERBOSE_FLAG:
            print(f'Instantiating {SIMULATION_TYPE}.')
        class_ = locate(f'src.Simulations.{SIMULATION_TYPE}.{SIMULATION_TYPE}')
        sim = class_(seed=SIMULATION_SEED,
                     iterations=SIMULATION_ITERATIONS,
                     cohort=cohort,
                     policy=policy,
                     error_log=error_log,
                     verbose=VERBOSE_FLAG,
                     **keyword_args)
    except:
        raise ValueError(f'Unable to instantiate Simulation {SIMULATION_TYPE}')

    actions, adherences, runtimes = sim.run()
    
    if LOG_SIMULATION_FLAG:
        DB_WRITER.add_simulation(SIM_ID, sim, actions, adherences, runtimes, 
                                 policy_init_time, "auto_id", "simulations")
        DB_WRITER.add_cohort(SIM_ID, cohort, "auto_id", "cohorts")
        DB_WRITER.add_policy(SIM_ID, SIM_ID, policy, "auto_id", "policies")

    return actions, adherences, cohort.transitions

if __name__ == "__main__":
    #### Initialize Parameters
    arg_groups = simutils.get_args(argv=None)
    
    config_filename = vars(arg_groups['general'])['config_file']
    print(config_filename)
    level = '.'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(os.path.join(level, config_filename))
    
    directories = simutils.setup_directories(config)
    
    params = OrderedDict()
    verbose = vars(arg_groups['general'])['verbose_flag'] 

    for group in ['cohort', 'policy']:
        params[group] = simutils.reconcile_args(group, 
                                                arg_groups[group], 
                                                config, 
                                                verbose=verbose)

    for group in ['general', 'simulation', 'database']:
        params[group] = simutils.reconcile_args(group, 
                                                arg_groups[group], 
                                                config, 
                                                verbose=verbose)[group]
    
    print(params['database'].items())
    db_writer = DBWriter(params['database'])

    if 'CPAPCohort' in params['cohort']:
        if 'BASIS_DIR' in params['cohort']['CPAPCohort']:
            print('Adjusting BASIS_DIR to be at the right level.')
            params['cohort']['CPAPCohort']['BASIS_DIR'] = os.path.join('..', 
                                                                       params['cohort']['CPAPCohort']['BASIS_DIR'])
    

    save_str = simutils.generate_savestr(**{k.lower(): v for k,v in params['general'].items()})
    if 'SIMULATION_ID' not in params['general']:
        # Then we assume that we are either running simulation once, or all repeats of simulation are listed in the config (and we are not overriding it with the command line args).
        # Either way, we are going to save the rows separately to combine later (if applicable)
        sim_id = get_foreign_key_vals.main(params['database'])
        n_repeats = len(params["cohort"].items())*len(params["policy"].items())
        
    else:
        sim_id = params['general']['SIMULATION_ID']
    
    ### Set up error logger:
    error_filepath = simutils.generate_filename(save_dir=directories['error_dir'], 
                                                save_keyword='error',
                                                save_str=save_str,
                                                save_type='log')
    simutils.setup_logger(logger_name='error_log',
                          log_file=error_filepath,
                          log_file_level=logging.INFO,
                          console_level=logging.WARNING)

    
    ### Run a simulation and save results
    for cohort, cohort_dict in params['cohort'].items():
        for policy, policy_dict in params['policy'].items():
            
            ## Run simulation
            print(f'Running a simulation with ID {sim_id}.')
            run_simulation(db_writer, 
                           sim_id, 
                           **params['general'], 
                           **cohort_dict, 
                           **policy_dict, 
                           **params['simulation'])
            
            
            ## Save results to files
            #  Deprecated. To use, specify logs_dir, actions_dir, adherences_dir, and arm_dir in the [paths] section of the config.
            
            # actions, adherences, transitions = run_simulation(db_writer,
            #                                                   sim_id,
            #                                                   **params['general'],
            #                                                   **cohort_dict, 
            #                                                   **policy_dict, 
            #                                                   **params['simulation'])
            
            # df = pd.DataFrame(data={**params['general'], 
            #                         **cohort_dict, 
            #                         **policy_dict, 
            #                         **params['simulation'], 
            #                         'adherences': f'{adherences.tolist()}', 
            #                         'actions': f'{actions.tolist()}'}, 
            #                   index=[sim_id])
            # save_filepath = simutils.generate_filename(save_dir=directories['logs_dir'],
            #                                            save_keyword='logs',
            #                                            save_str=save_str,
            #                                            save_type='csv')
            # df.to_csv(save_filepath)
            
            # ## Just to be extra cautious, let's save actions, adherences, and transition matrices
            # for save_type, arr in [('actions', actions),
            #                        ('adherences', adherences),
            #                        ('arm', transitions)]:
            #     if arr is not None:
            #         save_filepath = simutils.generate_filename(save_dir=directories[f'{save_type}_dir'],
            #                                                    save_keyword=save_type,
            #                                                    save_str=save_str,
            #                                                    save_type='npy')
            #         with open(save_filepath, 'wb') as f:
            #             np.save(f, arr)

            # # Free memory
            # actions = None
            # adherences = None
            # transitions = None
            
            if not params['database']['CLUSTER']:
                sim_id = get_foreign_key_vals.main(params['database'])

    db_writer.conn.close()
    
   
