import os
from collections import OrderedDict
import numpy as np
import pandas as pd

from src.Database.read import DBReader
import src.Analysis.compute_metrics as cm

def gen_no_fairness_vary_policy_df(reader: DBReader,
                                   policy_name_abbrs: dict,
                                   results_dir: str,
                                   exp_name: str = 'no_fairness_vary_policy',
                                   cohort_name: str = 'SyntheticCohort',
                                   sim_type: str = 'Simulation',
                                   n: int = 100,
                                   n_random: int = 100,
                                   k: int = 20,
                                   horizon: int = 180,
                                   sim_iterations: int = 100,
                                   local_reward: str = 'belief_identity',
                                   lb: float = 0.0,
                                   ub: float = 1.0,
                                   **kwargs):
    '''
    Generate results DataFrame by querying the database
    
    :param reader: DBReader connector to the database
    :type reader: DBReader
    :param policy_name_abbrs: abbreviations to use for the policy names
    :type policy_name_abbrs: dict
    :param results_dir: directory of the results
    :type results_dir: str
    :param exp_name: name of the experiment (collection of simulations), defaults to 'no_fairness_vary_policy'
    :type exp_name: str, optional
    :param cohort_name: name of the Cohort, defaults to 'SyntheticCohort'
    :type cohort_name: str, optional
    :param sim_type: name of the Simulation, defaults to 'Simulation'
    :type sim_type: str, optional
    :param n: number of arms in the cohort, defaults to 100
    :type n: int, optional
    :param n_random: number of random type arms, defaults to 100
    :type n_random: int, optional
    :param k: budget, defaults to 20
    :type k: int, optional
    :param horizon: simulation horizon, defaults to 180
    :type horizon: int, optional
    :param sim_iterations: number of iterations per simulation, defaults to 100
    :type sim_iterations: int, optional
    :param local_reward: name of the arms' local reward function, defaults to 'belief_identity'
    :type local_reward: str, optional
    :param lb: lower bound (probabilistic fairness), defaults to 0.0
    :type lb: float, optional
    :param ub: upper bound (probabilistic fairness), defaults to 1.0
    :type ub: float, optional
    :param **kwargs: unused kwargs
    :return: pd.DataFrame of experiment results
    
    Saves results to csv
    Saves LaTeX-formatted table of results

    '''
    if not os.path.exists(os.path.join(results_dir, exp_name)):
        os.mkdir(os.path.join(results_dir, exp_name))

    N = int(n)

    ## Query the db
    query = f""" SELECT sc.sim_id, sc.actions, sc.adherences,
                        p.policy_type, p.policy, p.lb, p.ub,
                        sc.cohort_type, sc.local_reward,
                        sc.n_arms, sc.n_random, sc.n_forward, sc.n_reverse, sc.n_concave, sc.n_convex,
                        sc.sim_type, sc.n_iterations,
                        p.k, p.horizon
                 FROM (SELECT s.auto_id AS sim_id,
                           s.actions, s.adherences,
                           c.auto_id AS cohort_id,
                           c.cohort_type, c.local_reward,
                           c.n_arms, c.n_random, c.n_forward, c.n_reverse, c.n_concave, c.n_convex,
                           s.sim_type, s.n_iterations
                       FROM simulations AS s, cohorts AS c
                       WHERE c.cohort_type = '{cohort_name}'
                           and c.n_arms = {N}
                           and c.n_random = {n_random}
                           and c.local_reward = '{local_reward}'
                           and s.sim_type = '{sim_type}'
                           and s.n_iterations = {sim_iterations}
                       ) AS sc
                 JOIN policies as p
                     ON (sc.sim_id = p.sim_id
                         AND sc.cohort_id = p.cohort_id)
                     WHERE p.k = {k}
                         and p.horizon = {horizon}
             """

    out = pd.read_sql_query(query, reader.conn)
    reader.cursor.close()

    # Filter lb and ub outside of the query
    prob_fair_res = out[(out['lb'] == lb) & (out['ub'] == ub)]
    other_policies_res = out[out['policy_type'] != "ProbFairPolicy"]
    res = pd.concat([prob_fair_res, other_policies_res])

    # Convert into numpy arrays (action, adherences, policy)
    res['action_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['actions']), 
                                dtype='int64').reshape(x['n_iterations'],
                                                       x['n_arms'],
                                                       x['horizon']), 
                                                       axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), 
                                dtype='int64').reshape(x['n_iterations'],
                                                       x['n_arms'],
                                                       x['horizon'] + 1),
        axis=1)
    res['policy_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['policy']), 
                                dtype='float64').reshape(x['n_arms']) 
        if x['policy'] is not None else None,
        axis=1)

    # Buckets and short names
    res['synth_cohort_subtype'] = res.apply(
        lambda row: cm.gen_synth_cohort_subtype(row[['n_random', 'n_forward', 'n_reverse', 'n_concave', 'n_convex']], 
                                                N),
        axis=1)
    res['policy_bucket'] = res.apply(lambda x: cm.map_policy_to_bucket(x['policy_type'], 
                                                                       x['sim_type']), 
                                     axis=1)
    res['policy_short_name'] = res.apply(lambda x: cm.policy_plus_param_names(policy_name_abbrs, 
                                                                              x['policy_type'],
                                                                              x['policy_bucket'], 
                                                                              None, 
                                                                              x['lb'], 
                                                                              None,
                                                                              x['local_reward'], 
                                                                              ub=x['ub'], ), 
                                         axis=1)

    # Drop Duplicates
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub',
                               'synth_cohort_subtype'], keep="last")

    # Compute reward
    res['local_rewards'] = res['adherence_arrays'].apply(cm.map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(cm.map_localr_to_R)

    # Get the baselines (keeping it general for future changes to the code)
    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()
    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") 
                                  & (res.sim_type == "Simulation") 
                                  & (res.local_reward == "belief_identity"), :]['action_arrays'].values


    for cohort_subtype in ['random']:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") 
                                                & (res.sim_type == "Simulation") 
                                                & (res.local_reward == "belief_identity") 
                                                & (res.synth_cohort_subtype == cohort_subtype), :][
            'global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") 
                                            & (res.sim_type == "Simulation") 
                                            & (res.local_reward == "belief_identity") 
                                            & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

    # Compute intervention benefit, wasserstein distance
    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_wd', 'sigma_wd']]))
    for simtype in res.sim_type.unique():
        for cohort_subtype in res.synth_cohort_subtype.unique():
            temp = res[(res.sim_type == simtype) & (res.synth_cohort_subtype == cohort_subtype)]

            tw_actions = temp.loc[(res.policy_type == "WhittleIndexPolicy") 
                                  & (temp.sim_type == "Simulation") 
                                  & (temp.local_reward == "belief_identity") 
                                  & (temp.synth_cohort_subtype == cohort_subtype), :]['action_arrays'].values
            first_action = tw_actions[0]
            ib_results = temp['global_rewards'].apply(
                lambda x: cm.compute_ib(no_act_R=no_act_R_vals[cohort_subtype][0], 
                                        tw_R=tw_R_vals[cohort_subtype][0],
                                        ref_alg_R=x, n_iter=sim_iterations))
            wd_results = temp.apply(lambda row: cm.compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                                round_robin_actions=round_robin_actions[0],
                                                                                tw_actions=first_action,
                                                                                normalize_wd_by_tw=True,
                                                                                take_exp_over_iters=True), 
                                    axis=1)
            temp['e_ib'] = [x[0] for x in ib_results]
            temp['sigma_ib'] = [x[1] for x in ib_results]
            temp['moe_ib'] = [x[2] for x in ib_results]
            temp['ci_ib'] = [x[3] for x in ib_results]
            temp['e_wd'] = [x[0] for x in wd_results]
            temp['sigma_wd'] = [x[1] for x in wd_results]
            temp['moe_wd'] = [x[2] for x in wd_results]
            temp['ci_wd'] = [x[3] for x in wd_results]

            out_df = out_df.append(temp)

    ## Construct filename and save df; LaTeX table
    res_filename = f'res_{exp_name}_{cohort_name}_N{N}_T{horizon}.csv'
    path = os.path.join(results_dir, exp_name, res_filename)
    cm.save_df_to_csv(out_df, path)

    tex_filename = f'tex_{exp_name}_{cohort_name}_N{N}_T{horizon}.txt'
    tex_path = os.path.join(results_dir, exp_name, tex_filename)
    cm.csv_to_latex_table(csv_path=path,
                       save_path=tex_path,
                       policy_buckets=['prob_fair', 'comparison'],
                       sort_order=['lb', 'ub', 'policy_type'],
                       sort_ascending=[False, True, False], size='small', percent_flag=[True, True])

    return res