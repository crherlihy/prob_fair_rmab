import logging
import os
import configparser
import argparse
from collections import OrderedDict


# FOR ERROR LOGGING
def setup_logger(logger_name, log_file, log_file_level=logging.INFO, console_level=logging.WARNING):
    """
    Initializes the logger (which saves simulation information).
    :param logger_name:
    :param log_file:
    :param log_file_level: defaults to logging.INFO
    :param console_level: defaults to logging.WARNING
    :return: None, updates in place

    """
    log_setup = logging.getLogger(logger_name)

    for handler in log_setup.handlers[:]:
        log_setup.removeHandler(handler)

    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', 
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='w+')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(console_level)
    log_setup.setLevel(log_file_level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)

def create_sub_dirs(dir_path):
    """
    Creates sub directories if they do not exist.
    :param dir_path: desired filepath (string)
    :return: None

    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            return

def setup_directories(config, depth='..'):
    '''
    Set up directories: create sub directories and return directory dict.
    
    :param config: configparser.ConfigParser() object that specifies run params
    :param depth: string indicating the relative depth from src. Default '..' puts the root on the same level as src.
    :return: dict of directories for saving/logging simulation results

    '''
    
    directories =  dict(config.items('paths'))
    for sub_dir in directories:
        directories[sub_dir] = os.path.join(depth, directories[sub_dir])
        # directories[sub_dir] = os.path.join(directories[sub_dir])
        create_sub_dirs(directories[sub_dir])
    
    return directories

def get_args(argv=None):
    '''
    
    :param argv: unused kwarg, defaults to None
    :return: dict of command-line args

    '''
    parser = argparse.ArgumentParser()
    general = parser.add_argument_group('general') # general configuration or control flow settings
    
    # Not in config:
    general.add_argument('-c', '--config_file', type=str, default='../config/example_simulations.ini', nargs='?',
                         help='relative filepath to config from root')
    general.add_argument('-eid', '--experiment_id', type=int, nargs='?',
                         help='Unique identifier of the experiment. If not set, no ID is used.')
    general.add_argument('-id', '--simulation_id', type=int, nargs='?',
                         help='Unique identifier of the simulation. If not set, no ID is used.')
    general.add_argument('-v', '--verbose_flag', type=bool, default=False, nargs='?',
                         help='print additional information to the console')
    
    # general.add_argument('--trial_name', type=str, nargs='?',
    #                      help='use in file and folder naming')
    # general.add_argument('--date', type=str, nargs='?',
    #                      help='use in file naming, format YYYYMMDD')
    general.add_argument('--log_simulation_flag', type=bool, nargs='?',
                         help='whether to save simulation parameters and results')

    # database
    database = parser.add_argument_group('database')
    database.add_argument('--db_ip', type=str, nargs='?', 
                          help="IP address of the node where the database is running (local IP if local; overrides config db ip parameter")

    
    # cohort 
    cohort = parser.add_argument_group('cohort')
    cohort.add_argument('--cohort_name', type=str, nargs='?',
                        help='run cohort [COHORT_NAME] only (overrides bool mappings contained in config parameter RUN_COHORT_FLAGS)')
    cohort.add_argument('--arm_type', type=str, nargs='?',
                        help='Arm class name (e.g. CollapsingArm)')
    cohort.add_argument('-n', '--n_arms', type=int, nargs='?',
                        help='number of arms in a cohort (N)')
    cohort.add_argument('--cohort_seed', type=int, nargs='?',
                        help='seed of cohort')
    cohort.add_argument('--local_reward', type=str, nargs='?',
                        help='string keyword for local reward')
    
    cohort.add_argument('--n_general', type=int, nargs='?',
                        help='number of general demographic arms')
    cohort.add_argument('--n_male', type=int, nargs='?',
                        help='number of male demographic arms')
    cohort.add_argument('--n_female', type=int, nargs='?',
                        help='number of female demographic arms')
    cohort.add_argument('--n_adhering', type=int, nargs='?',
                        help='number of `adhering` cluster arms, Kang et. al. 2013')
    cohort.add_argument('--n_nonadhering', type=int, nargs='?',
                        help='number of `non-adhering` cluster arms, Kang et. al. 2013')
    cohort.add_argument('--intervention_effect', type=float, nargs='+', # accepts multiple entries!
                        help='effect of an intervention')
    cohort.add_argument('--sigma', type=float, nargs='?',
                        help='logistic noise')
    
    cohort.add_argument('--n_forward', type=int, nargs='?',
                        help='number of forward threshold optimal arms')
    cohort.add_argument('--n_reverse', type=int, nargs='?',
                        help='number of reverse threshold optimal arms')
    cohort.add_argument('--n_concave', type=int, nargs='?',
                        help='number of concave arms, including forward')
    cohort.add_argument('--n_convex', type=int, nargs='?',
                        help='number of convex arms, including reverse')
    cohort.add_argument('--n_random', type=int, nargs='?',
                        help='number of random arms')
    cohort.add_argument('--override_threshold_optimality_conditions', type=bool, nargs='?',
                        help='whether to use belief_identity for threshold optimality')
    
    # policy
    policy = parser.add_argument_group('policy')
    policy.add_argument('--policy_name', type=str, nargs='?',
                        help='run policy [POLICY_NAME] only (overrides bool mappings contained in config parameter RUN_POLICY_FLAGS)')
    policy.add_argument('-k', type=int, nargs='?',
                        help='budget of arm pulls (k)')
    policy.add_argument('-T', '--horizon', type=int, nargs='?',
                        help='length of a simulation (T)')
    policy.add_argument('--interval_len', type=int, nargs='?',
                        help='length of an interval')
    
    policy.add_argument('--min_sel_frac', type=float, nargs='?',
                        help='minimum selection fraction per interval')
    policy.add_argument('--min_pull_per_pd', type=int, nargs='?',
                        help='minimum pulls per interval, MathProgPolicy only')
    
    policy.add_argument('--policy_seed', type=int, nargs='?',
                        help='seed of policy, for RandomPolicy or ProbFairPolicy')
    policy.add_argument('-lb', '--prob_pull_lower_bound', type=float, nargs='?',
                        help='probabisistic lower bound on times pulled')
    policy.add_argument('-ub', '--prob_pull_upper_bound', type=float, nargs='?',
                        help='probabisistic upper bound on times pulled')
    
    # simulation
    simulation = parser.add_argument_group('simulation')
    simulation.add_argument('--simulation_type', type=str, nargs='?',
                            help='run simulation [SIMULATION_TYPE]')

    simulation.add_argument('--simulation_seed', type=int, nargs='?',
                            help='seed of a simulation')
    simulation.add_argument('--simulation_iterations', type=int, nargs='?',
                            help='number of simulation iterations')
    
    simulation.add_argument('--heuristic', type=str, nargs='?',
                            help='type of heuristic, for IntervalHeuristicSimulation')
    
    
    args = parser.parse_args()
    #if vars(args)['verbose_flag'] == True:
     #   parser.print_help()
      #  print()
        
    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)
        
    return arg_groups

def reconcile_args(group: str, args: argparse.Namespace,
                   config: configparser.ConfigParser,
                   verbose: bool = False):
    '''
    Reconcile arguments from command-line args and config parameters
    
    Prioritization:
    1. args
    2. specific config settings (where applicable)
    3. default config settings 
    Therefore, builds dict from 3. to 1., replacing vals as it goes.
    
    There may be multiple cohort or policy groups, so a dict-of-dicts is returned for them.
    
    :param group: group name, e.g. 'cohort'
    :type group: str
    :param args: command-line arguments
    :type args: argparse.Namespace
    :param config: config parameters
    :type config: configparser.ConfigParser
    :param verbose: whether to print to the console, defaults to False
    :type verbose: bool, optional
    :return: OrderedDict (of OrderedDict) of params

    '''
    
    params = OrderedDict()
    
    if group in ['cohort', 'policy']:
        if vars(args)[f'{group}_name'] is not None:
            keys = [vars(args)[f'{group}_name']]
        else:
            flags = config[group][f'run_{group}_flags'].replace(',', ' ').split()
            names = config[group][f'{group}_names'].replace(',', ' ').split()
            assert(len(flags)==len(names)) # check that the length of RUN_{group}_FLAGS matches {group}_NAMES in the config
            keys = []
            for index, flag in enumerate(flags):
                if eval(flag) is True:
                    keys = keys + [names[index]]
    else:
        keys = [group] # I'd prefer to remove a level
    
    for key in keys:
        params[key] = OrderedDict(config.items(group)) # 3. default settings
        
        if group in ['cohort', 'policy']: # then there exist specific settings in config
            params[key][f'{group}_name'] = key # save the name (key) 
            params[key][f'{group}_type'] = key # if type isn't specified, take this as the default value
            for k, v in dict(config.items(f'{group}_{key}')).items():
                if k not in params[key].keys():
                    if verbose:
                        print(f'Adding new key value pair from specific config setting: {k.upper()}={v}')
                    params[key][k] = v
                elif v != params[key][k]:
                    if verbose:
                        print(f'Overriding default value {k.upper()}={params[key][k]} with specific setting in config {k.upper()}={v}')
                    params[key][k] = v # 2. specfic config settings
                else:
                    if verbose:
                        print(f'Specific setting in config {k.upper()}={v} matches default value {k.upper()}={params[key][k]}.')

        # Before we get the args from command line, let's convert the value type.
        # Don't like doing this but didn't find a better option :/
        for k, v in params[key].items():
            try:
                params[key][k] = eval(v)
            except(NameError, SyntaxError, TypeError):
                params[key][k] = v
  
        # 1. From command line
        for k, v in vars(args).items():
            if k!= 'config_file' and v is not None:
                if k not in params[key].keys():
                    if verbose:
                        print(f'Adding new key value pair from command line arg: {k.upper()}={v}')
                    params[key][k] = v
                elif v != params[key][k]:
                    if verbose:
                        print(f'Overriding config value {k.upper()}={params[key][k]} with command line arg {k.upper()}={v}')
                    params[key][k] = v 
                else:
                    if verbose:
                        print(f'Command line arg {k.upper()}={v} matches config value {k.upper()}={params[key][k]}.')
            
        # Prune values that are no longer needed:
        for pruned_key in ['log_flag', f'{group}_names', f'run_{group}_flags']:
            val=params[key].pop(pruned_key, None)
            if verbose:
                print(f'Removing {pruned_key.upper()}={val}.')
        
        # Done! Now we just make the params UPPER
        params[key] = {k.upper(): v for k, v in params[key].items()}
        if verbose:
            print()
            
    return params

def generate_savestr(trial_name: str, 
                     date: int, 
                     simulation_id: int = None, 
                     **kwargs):
    '''
    Helper function to generate filenames for saving data
    
    :param trial_name: name of trial (or experiment)
    :type trial_name: str
    :param date: date, generally YYYYMMDD
    :type date: int
    :param simulation_id: simulation id, defaults to None
    :type simulation_id: int, optional
    :param **kwargs: unused kwargs
    :return: filename (without file type or directory)
    :rtype: str

    '''
    save_str_base = f'{trial_name}_{date}'
    if simulation_id is None:
        print('No simulation_id passed into utils.generate_savestr()')
        return save_str_base
    else:
        return save_str_base + f'_ID{simulation_id}'


def generate_filename(save_dir: str, 
                      save_keyword: str, 
                      save_type: str, 
                      save_str: str = None, 
                      **kwargs):
    '''
    Generate a filename to save data
    
    :param save_dir: directory to save
    :type save_dir: str
    :param save_keyword: keyword to prepend
    :type save_keyword: str
    :param save_type: filetype, e.g. 'csv'
    :type save_type: str
    :param save_str: primary filename string, defaults to None
    :type save_str: str, optional
    :param **kwargs: unused kwargs
    :return: filename (with directory path)
    :rtype: str

    '''
    # save_type should not include the period, e.g. save_type = 'csv'
    if save_str is None:
        save_str = generate_savestr(**kwargs)
    return os.path.join(save_dir, f'{save_keyword}_{save_str}.{save_type}')
    
if __name__ == "__main__":
    pass
