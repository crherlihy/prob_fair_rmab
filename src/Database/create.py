'''create.py: create tables in the database
'''

import configparser
import os
from collections import OrderedDict
import mysql.connector as database

import src.utils as simutils

# def create_db_cmd(db_name):
#     return """CREATE DATABASE IF NOT EXISTS {}""".format(db_name)

def create_tables():
    '''
    Returns the SQL commands needed to create and link tables in the database
    
    :return: CREATE and ALTER SQL commands
    :rtype: (str)

    '''

    commands = (
    """
    CREATE TABLE `simulations` (
  `auto_id` int PRIMARY KEY,
  #`id` int,
  #`config_filename` varchar(255), 
  `sim_type` varchar(255),
  `heuristic_interval_len` int,
  `seed` int,
  `experiment_id` int, 
  `n_iterations` int,
  `error_log` varchar(255),
  `verbose` bool,
  `save_file_path` varchar(255),
  #`iter_seeds` longblob,
  `actions` longblob,
  `adherences` longblob,
  `heuristic` enum('first', 'last', 'random'),
  `interval_len` int,
  `runtimes` longblob,
  `policy_init_time` float)
    """,

    """
    CREATE TABLE `cohorts` (
  `sim_id` int,
  `auto_id` int PRIMARY KEY,
  `cohort_type` enum('SyntheticCohort', 'CPAPCohort'),
  `cohort_name` varchar(255),
  `seed` int,
  `arm_type` enum('RestlessArm', 'CollapsingArm'),
  `n_arms` int,
  `initial_state` int,
  `n_states` int,
  `n_actions` int,
  `pull_action` int,
  `local_reward` varchar(255),
  `lambd` float,
  `rho` mediumblob,
  `local_r` mediumblob,
  `global_R` mediumblob,
  `beta` float,
  `load_cohort` bool,
  `error_log` varchar(255),
  `verbose` bool,
  `transitions` mediumblob,
  `arms` mediumblob,
  `cohort_key` mediumblob,
  `n_forward` int,
  `n_reverse` int,
  `n_convex` int,
  `n_concave` int,
  `n_random` int,
  `override_threshold_optimality_conditions` bool,
  `n_male` int,
  `n_female` int,
  `n_adhering` int,
  `n_nonadhering` int,
  `truncate_adhering` bool,
  `truncate_nonadhering` bool,
  `basis_dir` varchar(255),
  `intervention_effect` mediumblob,
  `sigma` float)
    """,

    """
    CREATE TABLE `policies` (
  `auto_id` int PRIMARY KEY,
  `sim_id` int,
  `policy_type` enum('MathProgPolicy', 'MyopicPolicy', 'NoActPolicy', 'ProbFairPolicy', 'RandomPolicy', 'RoundRobinPolicy', 'WhittleIndexPolicy'),
  `cohort_id` int,
  `horizon` int,
  `k` int,
  `interval_len` int,
  `min_sel_frac` float,
  `min_pull_per_pd` int,
  `math_prog_type` enum('linear', 'integer'),
  `policy` mediumblob,
  `lb` float,
  `ub` float,
  `z` float,
  `epsilon` float,
  `flag_piecewise_linear_approx` bool,
  `ncut` int,
  `seed` int,
  `whittle_index_matrix` mediumblob)
    """,

    """ALTER
    TABLE
    `cohorts`
    ADD
    FOREIGN
    KEY(`sim_id`)
    REFERENCES
    `simulations`(`auto_id`)""",

    """ALTER
    TABLE
    `policies`
    ADD
    FOREIGN
    KEY(`sim_id`)
    REFERENCES
    `simulations`(`auto_id`)""",

    """ALTER
    TABLE
    `policies`
    ADD
    FOREIGN
    KEY(`cohort_id`)
    REFERENCES
    `cohorts`(`auto_id`)"""
    )

    return commands


def drop_tables():
    '''
    BE CAREFUL!! 
    DO NOT USE THIS UNLESS YOU INTEND TO DROP TABLES AND LOSE ALL CONTENT! 
    FOR DEBUGGING DURING SET-UP.
    
    :return: commands to drop all tables
    :rtype: (str)

    '''
    commands = ("""SET FOREIGN_KEY_CHECKS = 0""",
               """ DROP TABLE cohorts""", """DROP TABLE simulations""", """ DROP TABLE policies""")
    return commands


def delete_by_sim_exp_ids(sim_ids: [int]):
    '''
    Delete entries in the database by simulation ids
    
    :param sim_ids: simulation ids
    :type sim_ids: [int]
    :return: SQL commands to delete simulation entries
    :rtype: (str)

    '''

    commands = ("DELETE from policies where sim_id IN ({})""".format(', '.join(str(x) for x in sim_ids)),
                """DELETE from cohorts where sim_id IN ({})""".format(', '.join(str(x) for x in sim_ids)),
                """DELETE  from simulations where auto_id IN ({})""".format(', '.join(str(x) for x in sim_ids)))

    return commands


def main(db_params: OrderedDict, drop_all_tables: bool = False):
    '''
    Creates tables in the database
    
    :param db_params: database parameters to establish connection
    :type db_params: OrderedDict
    :param drop_all_tables: whether to drop all tables first, defaults to False
    :type drop_all_tables: bool, optional
    :return: None

    '''

    if db_params['database']['cluster']:

    # Establish database connection
        conn = database.connect(database=db_params['database']['db'], 
                                user=db_params['database']['user'], 
                                host=db_params['database']['host'], 
                                unix_socket="/tmp/mysql.sock")

    else:
    # Establish database connection
        conn = database.connect(db_params['database']['db'], 
                                user=db_params['database']['user'], 
                                host=db_params['database']['host'])

    conn.autocommit = True
    cursor = conn.cursor()    

    if drop_all_tables:
        sanity_check = input("Are you sure you want to drop all tables? Enter Y to confirm: ")
        if sanity_check == "Y":
            drop_cmds = drop_tables()
            for dcmd in drop_cmds:
                try:
                    cursor.execute(dcmd)
                except Exception as e:
                    print("Cannot drop table: {}".format(e))
            # assert cursor.fetchone() is None

        else:
            print("Sanity check failed; tables will not be dropped.")
            conn.close()
            return
    else:
        table_cmds = create_tables()
        
        for tcmd in table_cmds:
            try:
                cursor.execute(tcmd)

            except database.errors.DatabaseError as e:
                print("Cannot create table: {}".format(e))
                continue

    cursor.close()
    # Close the connection
    conn.close()
    return


if __name__ == "__main__":

    arg_groups = simutils.get_args(argv=None)
    config_filename = vars(arg_groups['general'])['config_file']
    level = '..'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(os.path.join(level, config_filename))

    main(OrderedDict(config), drop_all_tables=False)
