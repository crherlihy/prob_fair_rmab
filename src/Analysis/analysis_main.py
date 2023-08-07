import os
from collections import OrderedDict
import configparser
import src.utils as simutils
from src.Database.read import DBReader
import src.Analysis.analyze_fairness_vary_policy as afair
import src.Analysis.analyze_vary_cohort_synth as asynth
import src.Analysis.analyze_vary_cohort_cpap as ac
import src.Analysis.analyze_no_fairness_vary_policy as anofair


def main(config: configparser.ConfigParser) -> None:
    '''
    Run analysis on simulation results
    
    :param config: configuration settings
    :type config: configparser.ConfigParser
    :return: Saves results in-place
    :rtype: None

    '''

    policy_name_abbrs = {'NoAct': 'NoAct',
                         'Myopic': 'Myopic',
                         'MathProg': 'IP',
                         'ProbFair': 'PF',
                         'Random': 'Rand',
                         'RoundRobin': 'RR',
                         'WhittleIndex_belief_identity': 'TW',
                         'WhittleIndex_mate21_concave': 'TW-RA'}

    funcs = [afair.gen_fairness_vary_policy_df, asynth.gen_vary_cohort_composition_df,
             ac.gen_cpap_df, anofair.gen_no_fairness_vary_policy_df]

    experiments = [f'{func.__name__}'[4:-3] for func in funcs]

    print(experiments, funcs)
    for etype, f in zip(experiments, funcs):
        print(etype, f, list(config[etype].items()))
        f(reader, policy_name_abbrs=policy_name_abbrs, **config[etype])

    return


if __name__ == "__main__":
    arg_groups = simutils.get_args(argv=None)
    config_filename = "../config/example_analysis.ini"
    level = '..'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(os.path.join(level, config_filename))
    reader = DBReader(OrderedDict({k.upper(): v for k, v in config['database'].items()}))
    reader.cursor.execute('set global max_allowed_packet=67108864')

    main(config)

    reader.cursor.close()
    reader.conn.close()
