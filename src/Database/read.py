'''DBReader: database connector to read from the database
'''
import configparser
from collections import OrderedDict
import os
import zlib
import mysql.connector as database
import numpy as np
import pandas as pd

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

class DBReader(object):
    '''DBReader: database connector to read from the database
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
                                         host=db_params['HOST'], 
                                         unix_socket="/tmp/mysql.sock")

        else:
            # Establish database connection
            self.conn = database.connect(database=db_params['DB'], 
                                         user=db_params['USER'], 
                                         host=db_params['HOST'])

        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    @staticmethod
    def decompress_packet(compressed_packet):
        '''
        Decompress a compressed packet read from the database
        
        :param compressed_packet: compressed data (numpy arrays)
        :return: decompressed data

        '''
        return zlib.decompress(compressed_packet)
        
    def generic_query(self, eid: int):
        '''
        Generic query of the database to get results from a simulation ID
        
        :param eid: experiment (simulation) identifier
        :type eid: int
        :return: results
        :rtype: pd.DataFrame

        '''

        query = """SELECT sc.experiment_id, sc.sim_id, sc.cohort_id, p.policy_type, p.horizon, sc.actions, sc.adherences, sc.n_arms, sc.n_iterations from 
                (SELECT s.experiment_id, s.auto_id as sim_id, s.actions, s.adherences, s.n_iterations, c.auto_id as cohort_id, c.n_arms
                   FROM simulations as s, cohorts as c 
                   WHERE s.experiment_id = {} and s.auto_id = c.sim_id) as sc
                   JOIN policies as p
                   ON(sc.sim_id = p.sim_id and sc.cohort_id=p.cohort_id)""".format(eid)

        res = pd.read_sql_query(query, self.conn)

        res['action_arrays'] = res.apply(lambda x: np.frombuffer(self.decompress_packet(x['actions']), 
                                                                 dtype=int).reshape(x['n_iterations'], 
                                                                                    x['n_arms'], 
                                                                                    x['horizon']),
                                         axis=1)
        res['adherence_arrays'] = res.apply(lambda x: np.frombuffer(self.decompress_packet(x['adherences']), 
                                                                    dtype=int).reshape(x['n_iterations'], 
                                                                                       x['n_arms'], 
                                                                                       x['horizon']+1),
                                            axis=1)

        return res


def main(params: dict):
    '''
    Example usage of DBReader
    
    :param params: DBReader parameters
    :type params: dict
    :return: None

    '''
    reader = DBReader(params['database'])
    reader.generic_query(4)
    reader.cursor.close()

    # Close the connection
    reader.conn.close()
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
                          }

    params['config_file'] = config_filename
    main(params)
