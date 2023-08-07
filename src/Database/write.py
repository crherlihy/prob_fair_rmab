'''DBWriter: database connector to write to the database
'''

import configparser
from collections import OrderedDict
import os
from typing import Callable
import dill
import pickle
import zlib
import mysql.connector as database
import numpy as np

import src.utils as simutils
from src.Cohorts.Cohort import Cohort
from src.Cohorts.CPAPCohort import CPAPCohort
from src.Cohorts.SyntheticCohort import SyntheticCohort
from src.Simulations.Simulation import Simulation
from src.Simulations.IntervalHeuristicSimulation import IntervalHeuristicSimulation
from src.Policies.Policy import Policy
from src.Policies.MathProgPolicy import MathProgPolicy
from src.Policies.ProbFairPolicy import ProbFairPolicy
from src.Policies.NoActPolicy import NoActPolicy
from src.Policies.MyopicPolicy import MyopicPolicy
from src.Policies.RandomPolicy import RandomPolicy
from src.Policies.RoundRobinPolicy import RoundRobinPolicy
from src.Policies.WhittleIndexPolicy import WhittleIndexPolicy

class DBWriter(object):
    '''DBWriter: database connector to write to the database
    '''
    
    def __init__(self, db_params: OrderedDict):
        '''
        Initialize connection to the database
        
        :param db_params: database connection parameters
        :type db_params: OrderedDict
        :return: None

        '''

        if db_params['CLUSTER']:
            
            # Establish database connection
            self.conn = database.connect(database=db_params['DB'], 
                                         user=db_params['USER'], 
                                         host=db_params['HOST'])

        else:
            # Establish database connection
            self.conn = database.connect(database=db_params['DB'], 
                                         user=db_params['USER'], 
                                         host=db_params['HOST'])

        self.conn.autocommit = True
        self.cursor = self.conn.cursor()


    def convert_callable_to_str(self, f: Callable):
        '''
        Convert a callable function to string
        
        :param f: function
        :type f: Callable
        :return: string representation (from dill)

        '''
        return dill.dumps(f)

    @staticmethod
    def compress_packet(packet):
        '''
        Compress a packet of data (np.array)
        
        :param packet: data to compress
        :return: compressed data

        '''
        return zlib.compress(packet, 6)

    @staticmethod
    def decompress_packet(compressed_packet):
        '''
        Decompress a packet of data
        
        :param compressed_packet: compressed data
        :return: decopressed data (np.array)

        '''
        return zlib.decompress(compressed_packet)

    def add_experiment(self, id: int, config_filename: str) -> int:
        """
        Add an entry into the experiments table.
        Note: `auto_id` is the foreign key and it is auto-incremented upon insertion to ensure uniqueness.
        The user declares an experiment_id, which we're currently using to construct the experiment name.
        Thus, experiment_name does *not* have to currently be unique. 
        
        :param id: simulation identifier
        :param config_filename: name of config
        :returns: the auto_id associated with this experiment (so that we can use this value when writing to the simulations table)
        """
        self.cursor = self.conn.cursor()
        statement = """INSERT INTO experiments (auto_id, experiment_name, config_filename) VALUES (%s, %s, %s)"""
        data = (None, "experiment_{}".format(id), config_filename)
        self.cursor.execute(statement, data)
        self.cursor.close()

        return

    def add_simulation(self, 
                       sim_id: int, 
                       sim: Simulation, 
                       actions: np.array, 
                       adherences: np.array, 
                       runtimes: np.array, 
                       policy_init_time):
        '''
        Write to the simulations table
        
        :param sim_id: simulation id
        :type sim_id: int
        :param sim: Simulation to write
        :param actions: action data from simulating Simulation
        :type actions: np.array
        :param adherences: adherence data
        :type adherences: np.array
        :param runtimes: runtimes
        :type runtimes: np.array
        :param policy_init_time: policy initialization time
        :return: None

        '''

        self.cursor = self.conn.cursor()
       
        statement = """INSERT INTO simulations (auto_id, sim_type, seed, n_iterations,  error_log, 
                            verbose, save_file_path, actions, adherences, runtimes, policy_init_time, heuristic, heuristic_interval_len)
                             VALUES (%s, %s, %s, %s,%s, %s, %s,%s, %s, %s, %s, %s, %s)"""

        data = (sim_id, type(sim).__name__, sim.seed, sim.iterations, 
                sim.error_log.name, sim.verbose, None)

        if isinstance(sim,  IntervalHeuristicSimulation):
            data += (self.compress_packet(actions.tobytes()), 
                     self.compress_packet(adherences.tobytes()), 
                     self.compress_packet(runtimes.tobytes()),
                     None, sim.heuristic, sim.interval_len)
            print(f'Length of the simulation: {sim.interval_len}')

        elif isinstance(sim, Simulation):
            data += (self.compress_packet(actions.tobytes()), 
                     self.compress_packet(adherences.tobytes()), 
                     self.compress_packet(runtimes.tobytes()),
                     policy_init_time, None, None)
        
        self.cursor.execute('set global max_allowed_packet=1073741824')
        self.cursor.execute(statement, data)
        self.cursor.close()

        return

    def add_cohort(self, 
                   sim_id: int, 
                   cohort: Cohort):
        '''
        Write to the cohorts table
        
        :param sim_id: simulation identifier
        :type sim_id: int
        :param cohort: Cohort to write
        :return: None

        '''

        statement = """INSERT INTO cohorts (sim_id, auto_id, cohort_type, seed, arm_type, n_arms, initial_state,
        n_states, n_actions, pull_action, local_reward, lambd, rho, local_r, global_R, beta, cohort_name, load_cohort, error_log,
        verbose, transitions, arms, cohort_key, n_forward, n_reverse, n_convex, n_concave, n_random, override_threshold_optimality_conditions, 
        n_male, n_female, n_adhering, n_nonadhering, truncate_adhering, truncate_nonadhering, basis_dir, intervention_effect, sigma) 
        VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s,%s, %s, %s, %s, 
        %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s,%s, %s, %s, %s, 
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

        data = (sim_id, sim_id, type(cohort).__name__, cohort.seed, cohort.arm_type, cohort.n_arms, cohort.initial_state,
                     cohort.n_states, cohort.n_actions, cohort.pull_action, cohort.local_reward, cohort.lambd,
                     self.convert_callable_to_str(cohort.rho),self.convert_callable_to_str(cohort.r),
                     self.convert_callable_to_str(cohort.R), cohort.beta, cohort.cohort_name, cohort.load_cohort, cohort.error_log.name, cohort.verbose,
                     self.compress_packet(cohort.transitions.tobytes()), None, pickle.dumps(cohort.key))

        if isinstance(cohort, CPAPCohort):
            data += (None, None, None, None, None, None, cohort.n_male, cohort.n_female, cohort.n_adhering, cohort.n_nonadhering, cohort.truncate_adhering, cohort.truncate_nonadhering,
                     cohort.basis_dir, self.compress_packet(np.array(cohort.intervention_effect).tobytes()), cohort.sigma)

        elif isinstance(cohort, SyntheticCohort):
            data += (cohort.n_forward, cohort.n_reverse, cohort.n_convex, cohort.n_concave, cohort.n_random, cohort.override_threshold_optimality_conditions,
                     None, None, None, None, None, None, None, None, None)

        self.cursor = self.conn.cursor()
        self.cursor.execute('set global max_allowed_packet=1073741824')
        self.cursor.execute(statement, data)
        self.cursor.close()

        return

    def add_policy(self, 
                   sim_id: int, 
                   cohort_id: int, p: Policy):
        '''
        Write to the policies table
        
        :param sim_id: simulation identifier
        :type sim_id: int
        :param cohort_id: cohort identifier
        :type cohort_id: int
        :param p: Policy to write
        :return: DESCRIPTION
        :rtype: TYPE

        '''

        statement = """INSERT INTO policies (auto_id, sim_id, policy_type, cohort_id, horizon,
         k, interval_len, min_sel_frac, min_pull_per_pd,math_prog_type, policy, lb, ub, z, epsilon,
          flag_piecewise_linear_approx, ncut, seed, whittle_index_matrix) VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s,%s, %s, %s,%s, 
        %s, %s, %s, %s, %s, %s)"""

        data = (sim_id, sim_id, type(p).__name__, cohort_id, p.horizon, p.k)

        if isinstance(p, MathProgPolicy):
            data += (p.interval_len, p.min_sel_frac, p.min_pull_per_pd, p.type,
                     self.compress_packet(p.policy.tobytes()) if p.policy is not None else None, None, None, None, None, None, None, None, None)

        elif isinstance(p, (MyopicPolicy, NoActPolicy, RoundRobinPolicy)):
            data += (None, None, None, None, None, None, None, None, None, None, None, None, None)

        elif isinstance(p, ProbFairPolicy):
            data += (None, None, None, None, self.compress_packet(p.policy.tobytes()) if p.policy is not None else None, p.lb,
                     p.ub, float(p.z), p.epsilon, p.flag_piecewise_linear_approx, p.ncut if p.ncut is not None else None, p.seed, None)
        
        elif isinstance(p, RandomPolicy):
            data += (None, None, None, None, None, None, None, None, None, None, None, p.seed, None)

        elif isinstance(p, WhittleIndexPolicy):
            data += (None, None, None, None, None, None, None, None, None, None, None, None, self.compress_packet(p.whittle_index_matrix.tobytes()))

        self.cursor = self.conn.cursor()
        self.cursor.execute('set global max_allowed_packet=1073741824')
        self.cursor.execute(statement, data)
        self.cursor.close()

        return

    def insert_generic_row(self, 
                           table_name: str, 
                           row_info: dict, 
                           primary_key: str = "auto_id"):
        '''
        Example usage of writing to the database
        
        :param table_name: name of table
        :type table_name: str
        :param row_info: row data
        :type row_info: dict
        :param primary_key: primary key, defaults to "auto_id"
        :type primary_key: str, optional
        :return: key value
        :rtype: int

        '''
        statement = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name,', '.join(k for k in row_info.keys()),
                                                             ', '.join('%s' for _ in range(len(row_info.keys()))))
        data = tuple(row_info.values())
        self.cursor = self.conn.cursor()
        self.cursor.execute(statement, data)

        statement = """SELECT MAX({}) from {};""".format(primary_key, table_name)
        self.cursor.execute(statement)
        key_val = int(self.cursor.fetchone()[0])

        self.cursor.close()
        return key_val

def main(params: OrderedDict):
    '''
    Example usage of DBWriter
    
    :param params: database parameters
    :type params: OrderedDict
    :return: None

    '''
    writer = DBWriter(params['database'])
    
    writer.insert_generic_row("experiments",
                              {'auto_id': params['general']['EXPERIMENT_ID'], 
                               'experiment_name': str(0), 
                               'config_filename': params['config_file']})

    # Close the connection
    writer.conn.close()
    return


if __name__ == "__main__":
    arg_groups = simutils.get_args(argv=None)

    config_filename = vars(arg_groups['general'])['config_file']
    level = '..'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(os.path.join(level, config_filename))

    params = OrderedDict()
    params['database'] = {'DB': config['database']['db'],
                          'USER': config['database']['user'],
                          'HOST': config['database']['host'],
                          'CLUSTER': config['general']['cluster']}

    params['config_file'] = config_filename
    main(params)

