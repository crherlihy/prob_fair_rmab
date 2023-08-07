import os
import pandas as pd
import numpy as np
from collections import OrderedDict, Counter
from itertools import product
import configparser
import argparse
import src.utils as simutils
import glob
import seaborn as sns
from src.Cohorts.Cohort import Cohort
from src.Cohorts.CPAPCohort import CPAPCohort
from src.Database.read import DBReader
from typing import Type, Callable, Union
import matplotlib.pyplot as plt
import itertools
import zlib
import binascii
from scipy.stats import wasserstein_distance

def map_adherences_to_localr(adherences_tensor: np.array, local_r: Callable = lambda x: x) -> np.array:
    return np.apply_along_axis(local_r,-1, adherences_tensor)


def map_localr_to_R(localr_tensor: np.array, global_R: Callable = np.sum, avg_over_t: bool = False, time_dim: int = 2) -> np.array:
    if avg_over_t:
        T = localr_tensor.shape[time_dim]
        return np.apply_along_axis(lambda x: x*1/T, -1, np.apply_over_axes(global_R, localr_tensor, [1, 2]).ravel())
    else:
        return np.apply_over_axes(global_R, localr_tensor, [1,2]).ravel()


def compute_ib(no_act_R: np.array, tw_R: np.array, ref_alg_R: np.array) -> [float]:
    x =ref_alg_R - no_act_R
    y = tw_R - no_act_R
    z = [i == j for i,j in zip(x,y)]
    ibs = (ref_alg_R - no_act_R)/(tw_R - no_act_R)
    return np.mean(ibs), np.std(ibs)


def compute_pof(tw_R: np.array, ref_alg_R: np.array) -> [float]:
    pofs = (tw_R - ref_alg_R) / tw_R
    return np.mean(pofs), np.std(pofs)


def compute_hhi(actions_tensor: np.array, k: int, time_dim:int = 2, take_exp_over_iters: bool = True) -> [float]:
    T = actions_tensor.shape[time_dim]
    arm_action_sums = np.sum(actions_tensor, axis=time_dim)
    squared_avg_over_t = np.apply_along_axis(lambda x: (x*(1/(k*T)))**2, -1, arm_action_sums)

    if take_exp_over_iters:
        return np.mean(np.sum(squared_avg_over_t,axis=1)).ravel()[0], np.std(np.sum(squared_avg_over_t,axis=1)).ravel()[0]
    else:
        return np.sum(squared_avg_over_t,axis=1), None


def compute_wasserstein_distance(ref_alg_actions: np.array, round_robin_actions: np.array, take_exp_over_iters: bool = True, normalize_counts: bool = False) -> tuple:
    """
    Takes in two action array tensors (reference alg  and round robin); each tensor has dimension (n_iters, n_arms, horizon).
    For each simulation iteration, compute the pull_count cdf for each alg.
    Then, compute the first Wasserstein distance between these two 1D distributions:
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    If take_exp_over_iters evalautes to True, return mean and std_dev over all simulation iterations.
    Otherwise, return the array of Wasserstein distance values, of dimension (n_iters), and None

    @param ref_alg_actions: action tensor for reference algorithm; has dimensions (n_iterations, n_arms, horizon)
    @param round_robin_actions: action tensor for round robin algorithm; has dimensions (n_iterations, n_arms, horizon)
    @param take_exp_over_iters: boolean flag; indicates whether to compute E[wd] over simulation iterations.
    @param normalize_counts: boolean flag; indicates whether to normalize pull counts; defaults to False
    @return: (mean wd, sigma wd) if take_exp_over_iters evals to True, else: (wd_vals, None)
    """

    def convert_actions_to_pull_count_cdf(sim_actions: np.array, normalize: bool = False) -> np.array:
        """
        Helper function, takes in an action tensor with dimensions (n_arms, horizon)
        Counts the number of times each arm was pulled (e.g., over all timesteps, within a single simulation)
        @param sim_actions: action tensor with dimension (n_arms, horizon)
        @param normalize: boolean flag; indicates whether to normalize pull counts; defaults to False
        @return: cumulative distribution of pull counts, of dimension (n_iterations)
        """
        times_pulled = np.sum(sim_actions, axis=-1)
        unique_vals, counts = np.unique(times_pulled, return_counts=True)
        uvc_dict = {k:v for k,v in zip(unique_vals, counts)}

        # Not all possible cumulative values in {0...180} may be represented. We need a count for EACH possible value
        val_counts = {i:0 if i not in uvc_dict.keys() else int(uvc_dict[i]) for i in range(sim_actions.shape[-1]+2)}
        counts = list(val_counts.values())

        if normalize:
            counts = [x/sum(counts) for x in counts]
        return np.array(counts)

    def wasserstein(ref_counts: np.array, rr_counts: np.array) -> float:
        """
        @param ref_counts: 1xT array containing number of arms receiving each possible total pull count; reference alg
        @param rr_counts: 1xT array containing number of arms receiving each possible total pull count; RR alg
        @return: 1st Wasserstein distance metric (aka Earth mover's distance). assumes unit transport costs.
        """
        assert len(ref_counts) == len(rr_counts)
        diffs = ref_counts - rr_counts
        T = len(ref_counts)
        return sum([np.abs(sum(diffs[:i + 1])) for i in np.arange(T)])

    ref_pull_count_dists = [convert_actions_to_pull_count_cdf(x, normalize_counts) for x in ref_alg_actions]
    rr_pull_count_dists = [convert_actions_to_pull_count_cdf(x, normalize_counts) for x in round_robin_actions]

    wd_vals = [wasserstein(ref_alg, rr) for (ref_alg, rr) in zip(ref_pull_count_dists, rr_pull_count_dists)]

    if take_exp_over_iters:
        return np.mean(wd_vals), np.std(wd_vals)
    else:
        return wd_vals, None


def gen_synth_cohort_subtype(row, N: int):
    #print(row['n_forward'], row['n_reverse'])
    if row['n_forward'] == N:
        return 'forward'
    elif row['n_reverse'] == N:
        return 'reverse'
    elif row['n_random'] == N:
        return 'random'
    elif row['n_forward'] + row['n_reverse'] == N:
        return 'mixed'
    else:
        percent_convex = 100*row['n_convex']/N
        return f'{percent_convex:.0}% convex'

def map_policy_to_bucket(policy_type, sim_type=None):
    if policy_type == "ProbFairPolicy":
        return "prob_fair"
    elif policy_type in ['NoActPolicy', 'RandomPolicy', 'RoundRobinPolicy']:
        return "baseline"
    elif policy_type in "WhittleIndexPolicy" and sim_type == "Simulation":
        return "comparison"
    elif policy_type in "WhittleIndexPolicy" and sim_type == "IntervalHeuristicSimulation":
        return "heuristic"
    elif policy_type in ["MathProgPolicy"]:
        return "ip"

def policy_plus_param_names(abbr_dict: dict, policy_type: str, policy_bucket: str, heuristic: str,
                            ell: float, nu: float, local_r: str, min_sel_frac:float=None, min_pull_per_pd:float=None,
                            ub:float=None, ip_interval_len:int= None):
    # yikes sorry these args became unwieldy. kept catching cases late. TODO: refactor!
    if policy_bucket == "prob_fair":
        #return f"{abbr_dict['ProbFair']}\n" + r'$\ell$={}'.format(round(ell,2)) + "\n" + r'$u$={}'.format(round(ub,2))
        return f"{abbr_dict['ProbFair']}\n" + r'$\ell$={}'.format(round(ell, 2))
    elif policy_bucket == "heuristic":
        return r'H$_{}$'.format(heuristic[:1].upper()) + '\n' + r'$\nu$={} '.format(int(nu))
        #return r'H$_{}$'.format(heuristic[:4] if heuristic in ['last', 'random'] else heuristic[:5]) + '\n' + r'$\nu$={} '.format(int(nu))
    elif policy_bucket == "comparison":
        return abbr_dict["{}_{}".format(policy_type.replace("Policy", ""), local_r)]
    elif policy_bucket == "ip" and min_sel_frac == 0.0 and min_pull_per_pd == 0:
        return "{}\n".format(abbr_dict[policy_type.replace("Policy", "")]) + "baseline"
    elif policy_bucket == "ip":
        return "{}\n".format(abbr_dict[policy_type.replace("Policy", "")]) + r'$\psi$={} '.format(round(float(min_sel_frac),2)) if min_sel_frac != 0 else \
               "{}\n".format(abbr_dict[policy_type.replace("Policy", "")])+ r'$\nu$={} '.format(int(ip_interval_len))
    else:
        return abbr_dict[policy_type.replace("Policy", "")]

def compute_histogram_vals(df, n_iters, horizon, array_type='action'):

    def get_cumulative_pull_count_freqs(row):

        arr = row[f'{array_type}_arrays']
        # times_pulled has shape 500*100; contains values for total number of pulls each arm (N=100)
        # received in each simulation iteration (n_iters=500)
        times_pulled = np.sum(arr, axis=-1)

        # Unique vals = number of unique cumulative pull counts a given arm received over the simulation iterations
        # Counts = the number of time an arm received that number of cumulative pulls *over all simulation iterations*
        #unique_vals, counts = np.unique(times_pulled, return_counts=True, axis=-1) # orig. axis needs to be None so
        #   we don't get array counts
        unique_vals, counts = np.unique(times_pulled, return_counts=True)
        uvc_dict = {k:v for k,v in zip(unique_vals, counts)}

        # Not all possible cumulative values in {0...180} may be represented. We need to get a count for ALL possible values
        val_counts = {i:0 if i not in uvc_dict.keys() else int(uvc_dict[i]) for i in range(horizon+2)}

        data = {'vals': list(itertools.chain.from_iterable([np.repeat(k, v) for k,v in val_counts.items()])),
                'policy_short_name':row['policy_short_name'], 'policy_bucket': row['policy_bucket']}

        temp = pd.DataFrame.from_dict(data)
        return temp

    combined_df = pd.DataFrame(columns=['vals', 'policy_short_name', 'policy_bucket'])

    for i, row in df.iterrows():
        combined_df = combined_df.append(get_cumulative_pull_count_freqs(row))

    combined_df['vals'] = combined_df.vals.astype(int)

    return combined_df


def gen_cpap_df_old(reader: DBReader, results_dir: str, plots_dir:str, policy_name_abbrs: dict,  horizon:int, k:int = 20, interv_effect: float = 1.1,
                exp_name: str = "CPAP_experiments", n_iters:int = 500, make_plots:bool=True):
    """
    # 60 trials: (do we want to split into dif csvs based on these?)
    # --simulation_id
    # --intervention_effect
    # --horizon
    # --policy_name
    # --local_r
    # --simulation_name
    # --heuristic
    # --interval_len (else T)
    # --prob_pull_lower_bound (else -1.0)

    # NOT SURE WHETHER TO INCLUDE:
    # sc.local_r, sc.global_R (funcs)
    # cohort_key (not needed for CPAP)
    # sc.sigma, p.k (always the same val)

    Args:
        reader:
        results_dir:
        horizon:
        k:
        interv_effect:
        exp_name:

    Returns:

    """

    horizon = int(horizon)
    k = int(k)
    interv_effect = float(interv_effect)

    res_file_name = 'res_{}_interv{}_T{}.csv'.format(exp_name, int(interv_effect*100), horizon)
    tex_file_name = 'tex_{}_interv{}_T{}.txt'.format(exp_name, int(interv_effect*100), horizon)
    interv_effect = np.array([interv_effect, interv_effect])

    # Updated to incorporate removal of experiment table
    query = """ SELECT sc.sim_id, sc.sim_type, sc.intervention_effect, p.k, p.horizon,
                            p.policy_type, sc.cohort_type, sc.local_reward,  sc.heuristic, sc.heuristic_interval_len, 
                            p.lb, sc.actions, sc.adherences, sc.n_arms, sc.n_iterations
                FROM
                (SELECT s.auto_id AS sim_id,
                 s.actions, s.adherences, s.sim_type, s.heuristic, s.heuristic_interval_len,
                 c.n_arms, s.n_iterations, c.local_reward,
                 c.auto_id AS cohort_id, c.cohort_type, c.intervention_effect 
                 FROM simulations AS s, cohorts AS c 
                 WHERE c.cohort_type = 'CPAPCohort' and s.auto_id = c.sim_id) AS sc 
                JOIN policies AS p 
                ON (sc.sim_id = p.sim_id 
                    AND sc.cohort_id = p.cohort_id and p.k = {})
                WHERE p.horizon = {}""".format(k, horizon)

    res = pd.read_sql_query(query, reader.conn)

    reader.cursor.close()

    ## Format colns
    # I do not know how to filter on intervention effect inside the query because of the format
    res['interv'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['intervention_effect']), dtype=float), axis=1)
    res = res.loc[res.apply(lambda x: all(np.array(x.interv) == interv_effect), axis=1),:]

    res['action_arrays'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'], x['n_arms'], x['horizon']), axis=1)
    res['adherence_arrays'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'], x['n_arms'], x['horizon'] + 1),axis=1)


    res['policy_bucket'] = res.apply(lambda x: map_policy_to_bucket(x['policy_type'], x['sim_type']), axis=1)
    res['policy_short_name'] = res.apply(lambda x: policy_plus_param_names(policy_name_abbrs, x['policy_type'], x['policy_bucket'],
                                                                           x['heuristic'], x['lb'],
                                                                           x['heuristic_interval_len'],
                                                                           x['local_reward'],  x['ub'],),axis=1)

    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)
    res['avg_R'] = res['global_rewards'].apply(lambda x: np.mean(x))

    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_hhi', 'sigma_hhi', 'e_wd', 'sigma_wd']]))
    tw_R = res.loc[(res.policy_type == "WhittleIndexPolicy") & (res.sim_type == "Simulation") & (res.local_reward == "belief_identity"), :]['global_rewards'].values
    no_act_R = res.loc[(res.policy_type == "NoActPolicy") & (res.sim_type == "Simulation") & (res.local_reward == "belief_identity"), :]['global_rewards'].values

    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") & (res.sim_type == "Simulation") & (res.local_reward == "belief_identity"), :]['action_arrays'].values

    # Keep most recent run for each hyperparam combo
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'heuristic', 'heuristic_interval_len'],
                                  keep="last")

    for stype in res.sim_type.unique():

        temp = res[res.sim_type == stype]

        ib_results = temp['global_rewards'].apply(lambda x: compute_ib(no_act_R=no_act_R[0], tw_R=tw_R[0], ref_alg_R=x))
        hhi_results = temp['action_arrays'].apply(lambda x: compute_hhi(actions_tensor=x, k=k, time_dim=2, take_exp_over_iters=True))
        wd_results = temp.apply(lambda row: compute_wasserstein_distance(ref_alg_actions=row['action_arrays'], round_robin_actions=round_robin_actions[0],
                                                                                        take_exp_over_iters=True),axis=1)
        temp['e_ib'] = [x[0] for x in ib_results]
        temp['sigma_ib'] = [x[1] for x in ib_results]
        temp['e_hhi'] = [x[0] for x in hhi_results]
        temp['sigma_hhi'] = [x[1] for x in hhi_results]
        temp['e_wd'] = [x[0] for x in wd_results]
        temp['sigma_wd'] = [x[1] for x in wd_results]

        out_df = out_df.append(temp)

    if make_plots:
    # TODO: some plots are missing for T=30, need to also switch on that
    # yikes sorry so many things are hard-coded here  :( #TODO refactor
        for grp_name, plot_policy_group, policy_order in zip(["pf_bl_comp", "pf_heuristics"],
                                               [['prob_fair', 'baseline', 'comparison'], ['prob_fair', 'heuristic']],
                                               [['NoAct', 'Rand', 'RR', 'ProbFair\n$\\ell$=0.01', 'ProbFair\n$\\ell$=0.1',
                                                 'ProbFair\n$\\ell$=0.19', 'TW-RA', 'TW'],
                                                   [
                                                    'ProbFair\n$\\ell$=0.01','H$_F$\n$\\nu$=100 ', 'H$_L$\n$\\nu$=100 ','H$_R$\n$\\nu$=100 ',
                                                    'ProbFair\n$\\ell$=0.1','H$_F$\n$\\nu$=10 ', 'H$_L$\n$\\nu$=10 ','H$_R$\n$\\nu$=10 ',
                                                    'ProbFair\n$\\ell$=0.19', 'H$_F$\n$\\nu$=6 ', 'H$_L$\n$\\nu$=6 ','H$_R$\n$\\nu$=6 '
                                                    ]]):

            for plot_type, title, ytitle in zip(['action', 'adherence'], ["Arm-level Cumulative Pull Counts",
                                                                          "Arm-level Cumulative Adherence Counts"],
                                                ["Cumulative # of pulls received\n",
                                                 "Cumulative # of adherent timesteps\n"]):
                cdf = compute_histogram_vals(res, n_iters=n_iters, horizon=horizon, array_type=plot_type)
                temp = cdf[cdf.policy_bucket.isin(plot_policy_group)]

                plt.figure(figsize=(15, 10))
                g = sns.violinplot(x="policy_short_name", y="vals", hue="policy_short_name",
                                   data=temp, palette="Greys", split=False, dodge=False,
                                   scale="count", inner="quartiles", order=policy_order, hue_order=policy_order)

                g.legend_.remove()
                plt.ylabel(ytitle, fontsize=30, weight='bold')
                g.set(xlabel=None)
                plt.title("Distribution of " + title + "\n CPAP Cohort: intervention effect: {}; T = {}".format(interv_effect[0], horizon), fontsize=32, weight='bold')
                plt.yticks(fontsize=30, weight='bold')
                plt.xticks(fontsize=21)
                plt.setp(g.collections, alpha=.6)
                plt.savefig(os.path.join(plots_dir, "cpap_{}_T{}_{}_{}_dist_plot_pf.png".format(int(interv_effect[0]*100), horizon, grp_name, plot_type)),
                            bbox_inches='tight', )

    path = os.path.join(results_dir, exp_name, res_file_name)
    tex_path = os.path.join(results_dir, exp_name, tex_file_name)
    save_df_to_csv(out_df, path)
    csv_to_latex_table(csv_path=path, save_path=tex_path, # res_filter: dict = {'synth_cohort_subtype': 'random'},
                       policy_buckets=['prob_fair', 'heuristic', 'comparison', 'baseline'], 
                       metrics=['ib', 'hhi'], percent_flag=[True,False])
    return out_df

def gen_synthetic_df(reader: DBReader, results_dir: str, plots_dir: str,policy_name_abbrs:dict, cohort_type:str, cohort_name:str, n:int, horizon:int,
                     k:int = 20, exp_name: str = "synthetic_experiments", n_iters:int = 500, make_plots:bool = True) -> pd.DataFrame:
    """
    Split on cohort_name = random, forward, reverse, mixed
    AND horizon = 30, 180

    # 152 trials

    # --simulation_id
    # --cohort_name
    # --horizon
    # Policies list
    # --policy_name
    # TW policy fixed
    # --local_r
    # --simulation_name
    # --heuristic
    # --interval_len (else T)
    # ProbFair policy fixed
    # --prob_pull_lower_bound (else -1.0)

    # make sure to specify ub=1 OR null, else will also include vary_ub results
    # make sure to specify N, else could include ip results
    """

    N = int(n)
    horizon = int(horizon)
    k = int(k)

    res_file_name = "res_{}_".format(exp_name) + "{}_N{}_T{}.csv".format(cohort_name, str(N), str(horizon))
    tex_file_name = "tex_{}_".format(exp_name) + "{}_N{}_T{}.txt".format(cohort_name, str(N), str(horizon))

    # Updated to incorporate removal of experiment table
    query = """ SELECT sc.sim_id, sc.sim_type,  p.k, p.horizon,
                                p.policy_type, sc.cohort_type, sc.local_reward,  sc.heuristic, sc.heuristic_interval_len, p.interval_len,
                                p.lb, p.ub, sc.actions, sc.adherences, sc.n_arms, sc.n_iterations,sc.n_forward, sc.n_random, sc.n_reverse
                    FROM
                    (SELECT s.auto_id AS sim_id,
                     s.actions, s.adherences, s.sim_type, s.heuristic, s.heuristic_interval_len,
                     c.n_arms, s.n_iterations, c.local_reward,
                     c.auto_id AS cohort_id, c.cohort_type, c.intervention_effect, c.n_forward, c.n_reverse, c.n_random  
                     FROM simulations AS s, cohorts AS c 
                     WHERE c.cohort_type = '{}' and c.n_arms ={}) AS sc 
                    JOIN policies AS p 
                    ON (sc.sim_id = p.sim_id 
                        AND sc.cohort_id = p.cohort_id)
                    WHERE p.horizon = {} and p.k = {}""".format(cohort_type, N, horizon, k)

    out = pd.read_sql_query(query, reader.conn)
    reader.cursor.close()

    # make sure to specify ub=1 OR null, else will also include vary_ub results
    prob_fair_res = out[out.ub == 1.0]

    # But, for non-ProbFair policies, the value of ub is not meaningful and we want to keep these results
    other_policies_res = out[out.policy_type != "ProbFairPolicy"]
    res = pd.concat([prob_fair_res, other_policies_res])
    
    
    res['synth_cohort_subtype'] = res.apply(lambda row: gen_synth_cohort_subtype(row[['n_forward', 'n_random', 'n_reverse']], N),axis=1)
    res['action_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'],x['n_arms'], x['horizon']),axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'], x['n_arms'],
                                                                        x['horizon'] + 1), axis=1)

    res['policy_bucket'] = res.apply(lambda x: map_policy_to_bucket(x['policy_type'], x['sim_type']), axis=1)
    res['policy_short_name'] = res.apply(lambda x: policy_plus_param_names(policy_name_abbrs, x['policy_type'], x['policy_bucket'],
                                                                           x['heuristic'], x['lb'],
                                                                           x['heuristic_interval_len'],
                                                                           x['local_reward'],x['ub'],),axis=1)

    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)


    # # Keep most recent run for each hyperparam combo
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub', 'heuristic', 'heuristic_interval_len', 'synth_cohort_subtype'], keep="last")
    print('post drop', res.shape)

    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_hhi', 'sigma_hhi']]))

    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()
    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") & (res.sim_type == "Simulation") & (res.local_reward == "belief_identity"), :]['action_arrays'].values


    for cohort_subtype in ['forward', 'reverse', 'random', 'mixed']:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") &
                                                (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                                & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") &
                                                 (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                                 & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

    for simtype in res.sim_type.unique():
        for cohort_subtype in res.synth_cohort_subtype.unique():

            temp = res[(res.sim_type == simtype) & (res.synth_cohort_subtype == cohort_subtype)]

            ib_results = temp['global_rewards'].apply(lambda x: compute_ib(no_act_R=no_act_R_vals[cohort_subtype][0], tw_R=tw_R_vals[cohort_subtype][0], ref_alg_R=x))
            hhi_results = temp['action_arrays'].apply(lambda x: compute_hhi(actions_tensor=x, k=k, time_dim=2, take_exp_over_iters=True))

            wd_results = temp.apply(lambda row: compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                             round_robin_actions=round_robin_actions[0],
                                                                             take_exp_over_iters=True), axis=1)
            temp['e_ib'] = [x[0] for x in ib_results]
            temp['sigma_ib'] = [x[1] for x in ib_results]
            temp['e_hhi'] = [x[0] for x in hhi_results]
            temp['sigma_hhi'] = [x[1] for x in hhi_results]
            temp['e_wd'] = [x[0] for x in wd_results]
            temp['sigma_wd'] = [x[1] for x in wd_results]

            out_df = out_df.append(temp)

    if make_plots:

        for cohort_subtype in res.synth_cohort_subtype.unique():

            stype_df = res[(res.synth_cohort_subtype == cohort_subtype)]
            print(stype_df.policy_short_name.unique())

            # yikes sorry so many things are hard-coded here  :( #TODO refactor
            for grp_name, plot_policy_group, policy_order in zip(["pf_bl_comp", "pf_heuristics"],
                                                                 [['prob_fair', 'baseline', 'comparison'],
                                                                  ['prob_fair', 'heuristic']],
                                                                 [['NoAct', 'Rand', 'RR', 'ProbFair\n$\\ell$=0.01',
                                                                   'ProbFair\n$\\ell$=0.1',
                                                                   'ProbFair\n$\\ell$=0.19', 'TW-RA', 'TW'],
                                                                  ['ProbFair\n$\\ell$=0.01','H$_F$\n$\\nu$=100 ', 'H$_L$\n$\\nu$=100 ','H$_R$\n$\\nu$=100 ',
                                                                    'ProbFair\n$\\ell$=0.1','H$_F$\n$\\nu$=10 ', 'H$_L$\n$\\nu$=10 ','H$_R$\n$\\nu$=10 ',
                                                                    'ProbFair\n$\\ell$=0.19', 'H$_F$\n$\\nu$=6 ', 'H$_L$\n$\\nu$=6 ','H$_R$\n$\\nu$=6 ']]):

                for plot_type, title, ytitle in zip(['action', 'adherence'], ["Arm-level Cumulative Pull Counts",
                                                                              "Arm-level Cumulative Adherence Counts"],
                                                    ["Cumulative # of pulls received\n",
                                                     "Cumulative # of adherent timesteps\n"]):
                    cdf = compute_histogram_vals(stype_df, n_iters=n_iters, horizon=horizon, array_type=plot_type)
                    temp = cdf[cdf.policy_bucket.isin(plot_policy_group)]

                    plt.figure(figsize=(15, 10))
                    plt.rc('font', family='Times New Roman')
                    g = sns.violinplot(x="policy_short_name", y="vals", hue="policy_short_name",
                                       data=temp, palette="Greys", split=False, dodge=False,
                                       scale="count", inner="quartiles", order=policy_order, hue_order=policy_order)
                    g.set(xlabel=None)
                    g.legend_.remove()
                    plt.ylabel(ytitle, fontsize=30, weight='bold')

                    plt.title("Distribution of " + title + "\n Synthetic Cohort: type: {}; T = {}".format(cohort_subtype,horizon), fontsize=32, weight='bold')
                    plt.yticks(fontsize=30, weight='bold')
                    plt.xticks(fontsize=23)
                    plt.setp(g.collections, alpha=.6)
                    plt.savefig(os.path.join(plots_dir, "synth_{}_T{}_{}_{}_dist_plot_pf.png".format(cohort_subtype, horizon, grp_name, plot_type)),
                                bbox_inches='tight', )

    path = os.path.join(results_dir, exp_name, res_file_name)
    tex_path = os.path.join(results_dir, exp_name, tex_file_name)
    save_df_to_csv(out_df, path)
    csv_to_latex_table(csv_path=path, save_path=tex_path, # res_filter: dict = {'synth_cohort_subtype': 'random'},
                       policy_buckets=['prob_fair', 'heuristic', 'comparison', 'baseline'], 
                       metrics=['ib', 'hhi'], percent_flag=[True,False])
    return out_df


def gen_ip_df(reader:DBReader, results_dir: str, plots_dir:str, policy_name_abbrs:dict, n: int=2, horizon: int=6, k: int=1,
              exp_name: str = "IPvsProbFair_experiments",  n_iters:int = 500, make_plots:bool=True) -> pd.DataFrame:
    """
    Args:
        reader: DBReader object (to read from database)
        results_dir: directory to write experiment result file to
        plots_dir: directory to use when saving plots
        policy_name_abbrs: directory with policy long names as keys, and abbreviated names as values
        n: number of arms (default = 2)
        horizon: number of timesteps (default = 6)
        k: budget (default = 1)
        exp_name: experiment name (needs to correspond to a section in the results.ini config file)
        n_iters: number of simulation iterations (default = 500)
        make_plots: flag to determine whether to make plots (if false, only results csv is generated)

    Returns: pd.DataFrame containing empirical results for all synthetic cohorts
            Note: some cohort-hyperparam combinations may be run more than once; this function outputs the latest result per combo
    """
    N = int(n)
    horizon = int(horizon)
    k = int(k)

    # specify N!
    res_file_name = "res_{}_".format(exp_name) + "N{}_T{}.csv".format(str(N), str(horizon))
    tex_file_name = "tex_{}_".format(exp_name) + "N{}_T{}.txt".format(str(N), str(horizon))

    # Updated to incorporate removal of experiment_id
    query = """ SELECT sc.sim_id, p.k, p.horizon,
                                p.policy_type, sc.cohort_type, sc.local_reward, p.interval_len,
                                p.lb, p.ub, sc.actions, sc.adherences, sc.n_arms, sc.n_iterations,
                              p.min_sel_frac, p.min_pull_per_pd, sc.n_forward, sc.n_random, sc.n_reverse
                    FROM
                    (SELECT s.auto_id AS sim_id,
                     s.actions, s.adherences,
                     c.n_arms, s.n_iterations, c.local_reward,
                     c.auto_id AS cohort_id, c.cohort_type, c.intervention_effect, c.n_forward, c.n_reverse, c.n_random 
                     FROM simulations AS s, cohorts AS c 
                     WHERE c.cohort_type = 'SyntheticCohort' and c.n_random = {} and c.n_arms ={}) AS sc 
                    JOIN policies AS p 
                    ON (sc.sim_id = p.sim_id 
                        AND sc.cohort_id = p.cohort_id)
                    WHERE p.horizon = {} and p.k = {}""".format(N,N,horizon, k)

    res = pd.read_sql_query(query, reader.conn)
    reader.cursor.close()
    
    res['action_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'],x['n_arms'], x['horizon']),axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'], x['n_arms'],
                                                                        x['horizon'] + 1), axis=1)

    res['policy_bucket'] = res.apply(lambda x: map_policy_to_bucket(x['policy_type'], None), axis=1)
    res['policy_short_name'] = res.apply(lambda x: policy_plus_param_names(policy_name_abbrs, x['policy_type'], x['policy_bucket'],
                                                                           None, x['lb'], None,x['local_reward'],
                                                                           x['min_sel_frac'], x['min_pull_per_pd'], None,
                                                                           x['interval_len'], x['ub']),axis=1)
    # There's no TW baseline here, we're just going to save reward
    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)
    res['avg_R'] = res['global_rewards'].apply(lambda x: np.mean(x))

    # # Keep most recent run for each hyperparam combo
    res = res.drop_duplicates(['policy_short_name', 'interval_len', 'lb', 'ub', 'min_sel_frac', 'min_pull_per_pd'], keep="last")

    if make_plots:

        for plot_type, title, ytitle in zip(['action', 'adherence'], ["Arm-level Cumulative Pull Counts", "Arm-level Cumulative Adherence Counts"],
                                     ["Cumulative # of pulls received\n", "Cumulative # of adherent timesteps\n"]):

            cdf = compute_histogram_vals(res, n_iters=n_iters, horizon=horizon, array_type=plot_type)

            plt.figure(figsize=(15, 10))
            plt.rc('font', family='Times New Roman')
            g = sns.violinplot(x="policy_short_name", y="vals", hue="policy_short_name",
                             data=cdf, palette="binary", split=False, dodge=False,
                             scale="count", inner="quartiles",
                             hue_order=['IP\n$\\psi$=0.33 ', 'IP\n$\\nu$=3 ','ProbFair\n$\\ell$=0.33', 'ProbFair\n$\\ell$=0.0', 'IP\nbaseline' ],
                             order=['IP\n$\\psi$=0.33 ', 'IP\n$\\nu$=3 ','ProbFair\n$\\ell$=0.33', 'ProbFair\n$\\ell$=0.0', 'IP\nbaseline' ])

            g.legend_.remove()
            plt.ylabel(ytitle, fontsize=30, weight='bold')
            g.set(xlabel=None)
            plt.title("Distribution of "+ title, fontsize=32, weight='bold')
            plt.yticks(fontsize=30, weight='bold')
            plt.xticks(fontsize=30, weight='bold')
            plt.setp(g.collections, alpha=.6)
            plt.savefig(os.path.join(plots_dir, "{}_dist_plot_ip_vs_pf.png".format(plot_type)),bbox_inches='tight', )

    path = os.path.join(results_dir, exp_name, res_file_name)
    tex_path = os.path.join(results_dir, exp_name, tex_file_name)
    save_df_to_csv(res, path)
    csv_to_latex_table(csv_path=path, save_path=tex_path, # res_filter: dict = {'synth_cohort_subtype': 'random'},
                       policy_buckets=['prob_fair', 'heuristic', 'comparison', 'baseline'], 
                       metrics=['ib', 'hhi'], percent_flag=[True,False])
    return res
    
    
def gen_varyub_df(reader: DBReader, results_dir: str, policy_name_abbrs:dict, n:int, horizon:int, k:int = 20, exp_name: str = "varyub_experiments") -> pd.DataFrame:
    """
    # Split on cohort_name = random, forward, reverse, mixed
    # AND horizon = 30, 180

    # --simulation_id
    # --cohort_name
    # --horizon
    # -ub = 0.9

    # need to include from Synthetic as well.
    # pretty much the same as synthetic but ensuring lb = 0 OR N/A

    Args:
        reader:
        results_dir:
        cohort_name:
        N:
        horizon:
        k:
        exp_name:

    Returns:

    """

    N = int(n)
    horizon = int(horizon)
    k = int(k)

    res_file_name = "res_{}_".format(exp_name) + "allCohorts_N{}_T{}.csv".format(str(N), str(horizon))
    tex_file_name = "tex_{}_".format(exp_name) + "allCohorts_N{}_T{}.txt".format(str(N), str(horizon))

    # Updated to reflect removal of experiment_id
    query = """ SELECT sc.sim_id, sc.sim_type,  p.k, p.horizon,
                                p.policy_type, sc.cohort_type, sc.local_reward,  sc.heuristic, sc.heuristic_interval_len, p.interval_len,
                                p.lb, p.ub, sc.actions, sc.adherences, sc.n_arms, sc.n_iterations, sc.n_forward, sc.n_random, sc.n_reverse
                    FROM
                    (SELECT s.auto_id AS sim_id,
                     s.actions, s.adherences, s.sim_type, s.heuristic, s.heuristic_interval_len,
                     c.n_arms, s.n_iterations, c.local_reward,
                     c.auto_id AS cohort_id, c.cohort_type, c.intervention_effect,  c.n_forward, c.n_reverse, c.n_random  
                     FROM simulations AS s, cohorts AS c 
                     WHERE c.cohort_type = 'SyntheticCohort' and c.n_arms ={}) AS sc 
                    JOIN policies AS p 
                    ON (sc.sim_id = p.sim_id 
                        AND sc.cohort_id = p.cohort_id)
                    WHERE p.horizon = {} and p.k = {}""".format(N, horizon, k)

    # WHERE c.cohort_type = 'SyntheticCohort' and c.cohort_name = {} and c.n_arms ={}) AS sc
    out = pd.read_sql_query(query, reader.conn)
    reader.cursor.close()

    # make sure to specify ub=1 OR null, else will also include vary_ub results
    prob_fair_res = out[out.lb == 0.0]

    # But, for non-ProbFair policies, the value of ub is not meaningful and we want to keep these results
    other_policies_res = out[out.policy_type != "ProbFairPolicy"]
    res = pd.concat([prob_fair_res, other_policies_res])

    res['synth_cohort_subtype'] = res.apply(
        lambda row: gen_synth_cohort_subtype(row[['n_forward', 'n_random', 'n_reverse']], N), axis=1)

    res['action_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'],x['n_arms'], x['horizon']),axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'], x['n_arms'],
                                                                        x['horizon'] + 1), axis=1)

    #res['policy_short_name'] = res.apply(lambda x: gen_short_policy_names(x['policy_type'], {"lb": x['lb']}, None),axis=1)

    res['policy_bucket'] = res.apply(lambda x: map_policy_to_bucket(x['policy_type'], x['sim_type']), axis=1)
    res['policy_short_name'] = res.apply(lambda x: policy_plus_param_names(policy_name_abbrs, x['policy_type'], x['policy_bucket'],
                                                                           x['heuristic'], x['lb'],
                                                                           x['heuristic_interval_len'],x['local_reward'],
                                                                           None, None, x['ub']),axis=1)


    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)

    # # Keep most recent run for each hyperparam combo
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub', 'heuristic', 'heuristic_interval_len', 'synth_cohort_subtype'], keep="last")

    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_hhi', 'sigma_hhi']]))

    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()

    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") & (res.sim_type == "Simulation") & (res.local_reward == "belief_identity"), :]['action_arrays'].values


    for cohort_subtype in ['forward', 'reverse', 'random', 'mixed']:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") &
                                                (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                                & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") &
                                                 (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                                 & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

    for simtype in res.sim_type.unique():
        for cohort_subtype in res.synth_cohort_subtype.unique():

            temp = res[(res.sim_type == simtype) & (res.synth_cohort_subtype == cohort_subtype)]

            ib_results = temp['global_rewards'].apply(lambda x: compute_ib(no_act_R=no_act_R_vals[cohort_subtype][0], tw_R=tw_R_vals[cohort_subtype][0], ref_alg_R=x))
            hhi_results = temp['action_arrays'].apply(lambda x: compute_hhi(actions_tensor=x, k=k, time_dim=2, take_exp_over_iters=True))
            wd_results = temp.apply(lambda row: compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                             round_robin_actions=round_robin_actions[0],
                                                                             take_exp_over_iters=True), axis=1)

            temp['e_ib'] = [x[0] for x in ib_results]
            temp['sigma_ib'] = [x[1] for x in ib_results]
            temp['e_hhi'] = [x[0] for x in hhi_results]
            temp['sigma_hhi'] = [x[1] for x in hhi_results]
            temp['e_wd'] = [x[0] for x in wd_results]
            temp['sigma_wd'] = [x[1] for x in wd_results]


            out_df = out_df.append(temp)

    path = os.path.join(results_dir, exp_name, res_file_name)
    tex_path = os.path.join(results_dir, exp_name, tex_file_name)
    save_df_to_csv(out_df, path)
    csv_to_latex_table(csv_path=path, save_path=tex_path, # res_filter: dict = {'synth_cohort_subtype': 'random'},
                       policy_buckets=['prob_fair', 'heuristic', 'comparison', 'baseline'], 
                       metrics=['ib', 'hhi'], percent_flag=[True,False])
    return out_df

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
                                   lb = 0.0,
                                   ub = 1.0,
                                   **kwargs):
    ## Relevant fixed params:
    # sim_type = Simulation
    # cohort_name = SyntheticCohort
    # n_arms = 100
    # n_random = 100
    # k = 20
    # horizon = 180
    # simulation_iterations = 500
    # local_reward = belief_identity
    # lb = 0.0
    # ub = 1.0
    
    ## Relevant varying params:
    # policy_type
    
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
    other_policies_res = out[out.policy_type != "ProbFairPolicy"]
    res = pd.concat([prob_fair_res, other_policies_res])
    
    # Convert into numpy arrays (action, adherences, policy)
    res['action_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'],x['n_arms'], x['horizon']),axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'], x['n_arms'],
                                                                        x['horizon'] + 1), axis=1)
    res['policy_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['policy']), dtype='float64').reshape(x['n_arms']) if x['policy'] is not None else None, axis=1)
    
    # Buckets and short names
    res['synth_cohort_subtype'] = res.apply(
        lambda row: gen_synth_cohort_subtype(row[['n_random', 'n_forward', 'n_reverse', 'n_concave', 'n_convex']], N), axis=1)
    res['policy_bucket'] = res.apply(lambda x: map_policy_to_bucket(x['policy_type'], x['sim_type']), axis=1)
    res['policy_short_name'] = res.apply(lambda x: policy_plus_param_names(policy_name_abbrs, x['policy_type'],
                                          x['policy_bucket'], None, x['lb'],  None,
                                          x['local_reward'],ub=x['ub'],), axis=1)

    # Drop Duplicates
    # res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub', 'heuristic', 'heuristic_interval_len',
    #                            'synth_cohort_subtype'], keep="last")
    # print(res)
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub',
                               'synth_cohort_subtype'], keep="last")
    
    # Compute reward
    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)
    
    # Get the baselines (keeping it general for future changes to the code)
    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()
    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") & (res.sim_type == "Simulation") & (
                res.local_reward == "belief_identity"), :]['action_arrays'].values

    for cohort_subtype in ['random']:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") &
                                                (res.sim_type == "Simulation") & (res.local_reward == "belief_identity") &
                                                (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") &
                                            (res.sim_type == "Simulation") & (res.local_reward == "belief_identity") & 
                                            (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

    # print(res)
    # Compute intervention benefit, wasserstein distance
    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_wd', 'sigma_wd']]))
    for simtype in res.sim_type.unique():
        for cohort_subtype in res.synth_cohort_subtype.unique():
            temp = res[(res.sim_type == simtype) & (res.synth_cohort_subtype == cohort_subtype)]

            ib_results = temp['global_rewards'].apply(
                lambda x: compute_ib(no_act_R=no_act_R_vals[cohort_subtype][0], tw_R=tw_R_vals[cohort_subtype][0],
                                     ref_alg_R=x))
            wd_results = temp.apply(lambda row: compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                             round_robin_actions=round_robin_actions[0],
                                                                             take_exp_over_iters=True), axis=1)
            temp['e_ib'] = [x[0] for x in ib_results]
            temp['sigma_ib'] = [x[1] for x in ib_results]
            temp['e_wd'] = [x[0] for x in wd_results]
            temp['sigma_wd'] = [x[1] for x in wd_results]

            out_df = out_df.append(temp)
    # print(out_df)
    
    ## Plots
    # (currently none planned)
    
    ## Construct filename and save df; LaTeX table
    res_filename = f'res_{exp_name}_{cohort_name}_N{N}_T{horizon}.csv'
    path = os.path.join(results_dir, exp_name, res_filename)
    save_df_to_csv(out_df, path)
    
    tex_filename = f'tex_{exp_name}_{cohort_name}_N{N}_T{horizon}.txt'
    tex_path = os.path.join(results_dir, exp_name, tex_filename)
    csv_to_latex_table(csv_path=path, 
                       save_path=tex_path, 
                       policy_buckets=['prob_fair', 'comparison', 'baseline'], 
                       sort_order=['lb', 'ub', 'policy_type'],
                       sort_ascending=[False, True, False])
    
    return res


def gen_fairness_vary_policy_df(reader: DBReader,
                                policy_name_abbrs: dict,
                                results_dir: str,
                                plots_dir: str,
                                quantiles_file: str = 'PF_4quantiles_no_fairness_vary_policy_experiments_N100_k20.csv',
                                exp_name: str = 'fairness_vary_policy',
                                cohort_name: str = 'SyntheticCohort',
                                baselines: str = ['NoActPolicy', 'RoundRobinPolicy'],
                                n: int = 100,
                                n_random: int = 100,
                                k: int = 20,
                                horizon: int = 30,
                                sim_iterations: int = 100,
                                make_plots: bool = True,
                                **kwargs):

    def get_lb_ub_by_quantile(q_file: str, config_info: dict) -> pd.DataFrame:
        """
        Helper function to load quantiles df and exclude rows for which the corresponding nu value is not an int.
        @param q_file: Name of the quantiles file (directory is included in the config)
        @param config_info: Dictionary containing key-values from the config file (defaults to kwargs)
        @return: filtered df (one row per lb,ub,nu combo for which nu is an int, along with quantile/percentile info)
        """
        qdf = pd.read_csv(os.path.join(config_info['pf_quantiles_dir'], q_file))

        int_nu_combos = pd.DataFrame([row for _,row in qdf.iterrows() if row['nu'] == int(row['nu'])])
        lb_ub_nu_vals = pd.DataFrame(columns=['quantile', 'percentile', 'lb', 'ub', 'nu'])

        for i,row in int_nu_combos.iterrows():
            lb_ub_nu_vals= lb_ub_nu_vals.append({"quantile":row['quantile'], "percentile":row['percentile'], "lb":row['lb'], "ub":1.0, "nu":row['nu']},ignore_index=True)
            lb_ub_nu_vals = lb_ub_nu_vals.append({"quantile": row['quantile'], "percentile": row['percentile'], "lb": 0.0, "ub": row['ub'], "nu": row['nu']}, ignore_index=True)

        return lb_ub_nu_vals

    def merge_res_df_quantiles(qdf: pd.DataFrame, res_df: pd.DataFrame) -> pd.DataFrame:

        quantiles_dict = {round(q,5): i for i,q in enumerate(np.unique(qdf['quantile']))}
        nu_dict = {row['nu']: {'quantile': row['quantile'], 'percentile': row['percentile'],
                               'quantile_id': quantiles_dict[round(row['quantile'],5)]} for _,row in qdf.iterrows()}
        lbub_dict = {(round(row['lb'],5), round(row['ub'],5)): {'quantile': row['quantile'], 'percentile': row['percentile'],
                                                                'quantile_id':quantiles_dict[round(row['quantile'],5)]} for _, row in qdf.iterrows()}

        quantiles = np.zeros(res_df.shape[0])
        quantile_ids = np.zeros(res_df.shape[0])
        percentiles = np.zeros(res_df.shape[0])

        # myeh this is gross but merging was an even bigger pain
        for i,row in res_df.iterrows():
            if (round(row['lb'],5), round(row['ub'],5)) in lbub_dict:
                quantiles[i] = lbub_dict[(round(row['lb'],5), round(row['ub'],5))]['quantile']
                quantile_ids[i] = lbub_dict[(round(row['lb'], 5), round(row['ub'], 5))]['quantile_id']
                percentiles[i] = lbub_dict[(round(row['lb'],5), round(row['ub'],5))]['percentile']
            elif row['heuristic_interval_len'] in nu_dict:
                quantiles[i] = nu_dict[row['heuristic_interval_len']]['quantile']
                quantile_ids[i] = nu_dict[row['heuristic_interval_len']]['quantile_id']
                percentiles[i] = nu_dict[row['heuristic_interval_len']]['percentile']
            else:
                quantiles[i] = None
                quantile_ids[i] = None
                percentiles[i] = None

        res_df['quantile'] = quantiles
        res_df['quantile_id'] = quantile_ids
        res_df['percentile'] = percentiles

        return res_df

    N = int(n)
    res_file_name = "res_{}_".format(exp_name) + "{}_N{}_T{}.csv".format(cohort_name, str(N), str(horizon))
    tex_file_name = "tex_{}_".format(exp_name) + "{}_N{}_T{}.txt".format(cohort_name, str(N), str(horizon))

    if not os.path.exists(os.path.join(results_dir, exp_name)):
        os.mkdir(os.path.join(results_dir, exp_name))

    lb_ub_nu = get_lb_ub_by_quantile(quantiles_file, kwargs)
    print(lb_ub_nu)

    ## Query the db
    query = f""" SELECT sc.sim_id, sc.actions, sc.adherences,
                        p.policy_type, p.policy, p.lb, p.ub, 
                        sc.cohort_type, sc.n_arms, sc.n_random, sc.n_forward, sc.n_reverse, sc.local_reward,
                        sc.sim_type, sc.n_iterations, sc.heuristic, sc.heuristic_interval_len,
                        p.k, p.horizon
                 FROM (SELECT s.auto_id AS sim_id,
                           s.actions, s.adherences,
                           c.auto_id AS cohort_id,
                           c.cohort_type, c.n_arms, c.n_random, c.n_forward, c.n_reverse, c.local_reward,
                           s.sim_type, s.n_iterations, s.heuristic, s.heuristic_interval_len
                       FROM simulations AS s, cohorts AS c
                       WHERE c.cohort_type = '{cohort_name}'
                           and c.n_arms = {N}
                           and c.n_random = {n_random}
                           and s.n_iterations = {sim_iterations}
                       ) AS sc
                 JOIN policies as p
                     ON (sc.sim_id = p.sim_id
                         AND sc.cohort_id = p.cohort_id)
                     WHERE p.k = {k}
                         and p.horizon = {horizon}
             """

    res = pd.read_sql_query(query, reader.conn)
    reader.cursor.close()
    
    # Decompress action and adherence arrays
    res['action_arrays'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'],
                                                                                               x['n_arms'],
                                                                                               x['horizon']), axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'],
                                                                                                  x['n_arms'],
                                                                                               x['horizon'] + 1), axis=1)
    res['policy_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['policy']), dtype='float64').reshape(x['n_arms']) if x['policy'] is not None else None, axis=1)
    
    # Generate cohort subtype, policy bucket, and policy short name strings
    res['synth_cohort_subtype'] = res.apply(
        lambda row: gen_synth_cohort_subtype(row[['n_random', 'n_forward', 'n_reverse']], N), axis=1)
    res['policy_bucket'] = res.apply(lambda x: map_policy_to_bucket(x['policy_type'], x['sim_type']), axis=1)
    res['policy_short_name'] = res.apply(lambda x: policy_plus_param_names(policy_name_abbrs, x['policy_type'],
                                          x['policy_bucket'], x['heuristic'], x['lb'],  x['heuristic_interval_len'],
                                          x['local_reward'],ub=x['ub'],), axis=1)

    # Compute local and global rewards based on adherence information
    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)

    # # Keep most recent run for each hyperparam combo
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub', 'heuristic', 'heuristic_interval_len',
                               'synth_cohort_subtype'], keep="last")

    # Incorporate quantile/percentile information for ProbFair (lb,ub) + heuristic policies (heuristic_interval_len)
    #res = merge_res_df_quantiles(lb_ub_nu, res)
    #print(res[['policy_short_name', 'lb', 'ub', 'heuristic', 'heuristic_interval_len', 'quantile', 'percentile']])

    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_wd', 'sigma_wd']]))

    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()
    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") & (res.sim_type == "Simulation") & (
                res.local_reward == "belief_identity"), :]['action_arrays'].values

    for cohort_subtype in ['forward', 'reverse', 'random', 'mixed']:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") &
                                                (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                                & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") &
                                            (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                            & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

    for simtype in res.sim_type.unique():
        for cohort_subtype in res.synth_cohort_subtype.unique():
            temp = res[(res.sim_type == simtype) & (res.synth_cohort_subtype == cohort_subtype)]

            ib_results = temp['global_rewards'].apply(
                lambda x: compute_ib(no_act_R=no_act_R_vals[cohort_subtype][0], tw_R=tw_R_vals[cohort_subtype][0],
                                     ref_alg_R=x))

            wd_results = temp.apply(lambda row: compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                             round_robin_actions=round_robin_actions[0],
                                                                             take_exp_over_iters=True), axis=1)
            temp['e_ib'] = [x[0] for x in ib_results]
            temp['sigma_ib'] = [x[1] for x in ib_results]
            temp['e_wd'] = [x[0] for x in wd_results]
            temp['sigma_wd'] = [x[1] for x in wd_results]

            out_df = out_df.append(temp)

    if make_plots:

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        #temp = out_df[out_df.quantile_id.isin([0,1,2])]
        #print(temp.policy_short_name.unique())

        # Remove any ProbFair results where ub != 1
        temp = out_df[out_df.ub.isin([np.nan, 1.0])]

        no_act = out_df[out_df.policy_type == "NoActPolicy"]
        rr = out_df[out_df.policy_type == "RoundRobinPolicy"]
        ra_tw = out_df[(out_df.policy_type == "WhittleIndexPolicy") & (out_df.local_reward == "mate21_concave")]
        tw = out_df[(out_df.policy_type == "WhittleIndexPolicy") & (out_df.local_reward == "belief_identity") & (out_df.policy_bucket == "comparison")]

        #temp['percentile'] = temp['percentile'].apply(lambda x: round(x,2))
        #print(temp[['percentile', 'policy_short_name']].groupby(['percentile']).head())
        
        ## Bar plot
        for dep_var in ["ib", "wd"]:

            g = None

            if dep_var == "ib":
                temp = temp[(temp.policy_type != "NoActPolicy") & (temp.policy_bucket != "comparison")]
                policy_order = ['Rand', 'RR', 'PF\n$\\ell$=0.06', 'H$_F$\n$\\nu$=18 ', 'H$_L$\n$\\nu$=18 ', 'H$_R$\n$\\nu$=18 ',
                                    'PF\n$\\ell$=0.1', 'H$_F$\n$\\nu$=10 ', 'H$_L$\n$\\nu$=10 ', 'H$_R$\n$\\nu$=10 ',
                                    'PF\n$\\ell$=0.17', 'H$_F$\n$\\nu$=6 ', 'H$_L$\n$\\nu$=6 ', 'H$_R$\n$\\nu$=6 ']
            elif dep_var == "wd":
                temp = temp[(temp.policy_type != "RoundRobinPolicy") & (temp.policy_bucket != "comparison")]
                policy_order = ['Rand', 'PF\n$\\ell$=0.06', 'H$_F$\n$\\nu$=18 ', 'H$_L$\n$\\nu$=18 ', 'H$_R$\n$\\nu$=18 ',
                                    'PF\n$\\ell$=0.1', 'H$_F$\n$\\nu$=10 ', 'H$_L$\n$\\nu$=10 ', 'H$_R$\n$\\nu$=10 ',
                                    'PF\n$\\ell$=0.17', 'H$_F$\n$\\nu$=6 ', 'H$_L$\n$\\nu$=6 ', 'H$_R$\n$\\nu$=6 ' ]

            g = sns.barplot(data=temp, x="policy_short_name", y ="e_{}".format(dep_var),hue='policy_short_name',
                            dodge=False, palette='gray',alpha=0.7,
                            order=policy_order,)



            # plt.errorbar("policy_short_name", "e_{}".format(dep_var), "sigma_{}".format(dep_var),  solid_capstyle='projecting', capsize=3,
            #       color="blue",fmt='none')

            # g = sns.FacetGrid(temp, col="policy_type", aspect=1.2, sharex=False)
            # #g = sns.FacetGrid(temp, col="percentile",  aspect=1.2,sharex=False)

            # TODO: formatting of error bars; ordering of items; legend placement; numeric value labels; dep var names
            # g.map_dataframe(sns.barplot, x="policy_short_name", y="e_{}".format(dep_var),
            #                 hue='policy_short_name', dodge=False, palette='gray',alpha=0.7)
            #g.map(plt.errorbar, "policy_short_name", "e_{}".format(dep_var), "sigma_{}".format(dep_var),  solid_capstyle='projecting', capsize=3,
               #   color="blue",fmt='none')

            #g.set_ylabels("{}".format(dep_var))
            #g.set_xlabels("")
            #g.get_legend().set_visible(False)

            # for text in g.get_legend().texts:
            #     #if (text.get_text().): text.set_text('b')  # change label text
            #     text.set_visible(False)
            #
            # handles, labels = g.get_legend_handles_labels()
            # print(handles, labels)
            #ax.legend(handles, labels)
            #g.legend =

            #print(g.get_legend_handles_labels())

            if dep_var == "ib":
                g.axhline(y=rr["e_{}".format(dep_var)].median(), label='Round Robin {}'.format(dep_var), color='green')
                g.axhline(y=ra_tw["e_{}".format(dep_var)].median(), label='Risk-Aware TW {}'.format(dep_var),
                          color='blue')
                g.axhline(y=tw["e_{}".format(dep_var)].median(), label='TW {}'.format(dep_var),
                          color='red')

            elif dep_var == "wd":
                g.axhline(y=no_act["e_{}".format(dep_var)].median(), label='No Act {}'.format(dep_var),
                      color='orange')
                g.axhline(y=ra_tw["e_{}".format(dep_var)].median(), label='Risk-Aware TW {}'.format(dep_var),
                          color='blue')
                g.axhline(y=tw["e_{}".format(dep_var)].median(), label='TW {}'.format(dep_var),
                          color='red')

            print(g.get_legend_handles_labels())

            #labels = ["some name", "some other name", "horizontal"]
            handles, labels = g.get_legend_handles_labels()

            # Slice list to remove first handle
            g.legend(handles=handles[:3], labels=labels[:3])

            print(g.get_legend_handles_labels())

           # disable label
            #g.refline(y=rr["e_{}".format(dep_var)].median(), label='Round Robin {}'.format(dep_var), color='green')
            #g.refline(y=ra_tw["e_{}".format(dep_var)].median(), label='Risk-Aware TW {}'.format(dep_var), color='orange')
            # This is a hack bc we don't need to repeat the policy name->color mappings
            #temp = {k: v for k,v in g._legend_data.items() if k in ['Round Robin IB', 'Risk-Aware TW IB']}
            #print(g.axes.legend_)
            # g.axes.legend_ = {k: v for k,v in g.axes.legend_.values() if k in ['Round Robin {}'.format(dep_var),
            #                                                                'Risk-Aware TW {}'.format(dep_var)]}
            #

            #g._legend_data = {k: v for k,v in g._legend_data.items() if k in ['Round Robin {}'.format(dep_var),
                                                                           #'Risk-Aware TW {}'.format(dep_var)]}
            #g.add_legend()
            #plt.show()
            #g.get_legend().set_visible(True)
            g.figure.savefig(os.path.join(plots_dir, "{}_bar_plot_fairness_vary_policy.png".format(dep_var)),
                        bbox_inches='tight')
            plt.close()


            # TODO: save plot
            
        ## Violin plot (TODO)
    

    save_path = os.path.join(results_dir, exp_name, res_file_name)
    save_df_to_csv(out_df[out_df.ub.isin([np.nan, 1.0]) & (out_df.lb != 0)], save_path=save_path)
    tex_path = os.path.join(results_dir, exp_name, tex_file_name)
    csv_to_latex_table(csv_path=save_path,
                       save_path=tex_path,
                       res_filter={'synth_cohort_subtype': 'random'},
                       policy_buckets=['prob_fair', 'heuristic', 'comparison', 'baseline'],
                       metrics=['ib', 'wd'],
                       percent_flag=[True,False])

    return out_df

def gen_vary_cohort_composition_df(reader: DBReader, 
                                   policy_name_abbrs: dict,
                                   exp_name: str = 'vary_cohort_composition',
                                   cohort_name: str = 'SyntheticCohort',
                                   n: int = 100,
                                   k: int = 20,
                                   horizon: int = 180,
                                   sim_iterations: int = 100,
                                   lb: float = 0.1,
                                   ub: float = 1.0,
                                   inerval_len: int = 10, 
                                   **kwargs):
    ## Relevant fixed params:
    # cohort_name = SyntheticCohort
    # n_arms = 100
    # k = 20
    # horizon = 180
    # simulation_iterations = 500
    # lb = 0.1
    # ub = 1.0
    # interval_len = 10
    
    ## Relevant varying params:
    # policy_type 
    # percent_convex = 0, 25, 50, 75, 100
        # n_convex, n_concave
    
    
    N = int(n)
 
    ## Query the db
    query = f""" SELECT sc.sim_id, sc.actions, sc.adherences,
                        p.policy_type, p.policy, p.lb, p.ub,
                        sc.cohort_type, sc.n_arms, sc.n_forward, sc.n_reverse, sc.n_concave, sc.n_convex, sc.n_random, sc.local_reward,
                        sc.sim_type, sc.n_iterations,
                        p.k, p.horizon
                 FROM (SELECT s.auto_id AS sim_id,
                           s.actions, s.adherences,
                           c.auto_id AS cohort_id,
                           c.cohort_type, c.n_arms, c.n_forward, c.n_reverse, c.n_concave, c.n_convex, c.n_random, c.local_reward,
                           s.sim_type, s.n_iterations
                       FROM simulations AS s, cohorts AS c
                       WHERE c.cohort_type = '{cohort_name}'
                           and c.n_arms = {N}
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
    other_policies_res = out[out.policy_type != "ProbFairPolicy"]
    res = pd.concat([prob_fair_res, other_policies_res])
    
    # Convert into numpy arrays (action, adherences, policy)
    res['action_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'],x['n_arms'], x['horizon']),axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'], x['n_arms'],
                                                                        x['horizon'] + 1), axis=1)
    res['policy_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['policy']), dtype='float64').reshape(x['n_arms']) if x['policy'] is not None else None, axis=1)
    
    # Buckets and short names
    res['synth_cohort_subtype'] = res.apply(
        lambda row: gen_synth_cohort_subtype(row[['n_random', 'n_forward', 'n_reverse', 'n_concave', 'n_convex']], N), axis=1)
    res['policy_bucket'] = res.apply(lambda x: map_policy_to_bucket(x['policy_type'], x['sim_type']), axis=1)
    res['policy_short_name'] = res.apply(lambda x: policy_plus_param_names(policy_name_abbrs, x['policy_type'],
                                          x['policy_bucket'], x['heuristic'], x['lb'],  x['heuristic_interval_len'],
                                          x['local_reward'],ub=x['ub'],), axis=1)
    
    # Drop Duplicates
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub', 'heuristic', 'heuristic_interval_len',
                               'synth_cohort_subtype'], keep="last")
    
    # Compute reward
    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)
    
    # Compute intervention benefit
    
    # Compute wasserstein distance
    
    return res # for debugging

    ## Plots
    
    ## Construct filename and save df; LaTeX table
    # res_filename = f'res_{exp_name}_{cohort_name}_N{N}_T{horizon}.csv'
    # path = os.path.join(results_dir, exp_name, res_filename)
    # save_df_to_csv(out_df, path)
    
    # tex_filename = f'tex_{exp_name}_{cohort_name}_N{N}_T{horizon}.txt'
    # tex_path = os.path.join(results_dir, exp_name, tex_filename)
    # TODO: call csv_to_latex_table here.
    
    return res

def gen_cpap_df(reader: DBReader, 
                exp_name: str = 'cpap',
                cohort_name: str = 'CPAPCohort', 
                intervention_effect: float = 1.1,
                n: int = 100,
                k: int = 20,
                horizon: int = 180,
                sim_iterations: int = 100,
                lb: float = 0.1,
                ub: float = 1.0,
                interval_len: int = 10,
                **kwargs):
    ## Relevant fixed params:
    # sim_type = Simulation
    # cohort_name = CPAPCohort
    # intervention_effect = 1.1
    # n_arms = 100
    # k = 20
    # horizon = 180
    # simulation_iterations = 500
    # lb = 0.1
    # ub = 1.0
    # interval_len = 10
    
    ## Relevant varying params:
    # policy_type
    
    N = int(n)
 
    ## Query the db
    query = f""" SELECT sc.sim_id, sc.actions, sc.adherences,
                        p.policy_type, p.policy, p.lb, p.ub,
                        sc.cohort_type, sc.n_arms,
                        sc.sim_type, sc.n_iterations,
                        p.k, p.horizon
                 FROM (SELECT s.auto_id AS sim_id,
                           s.actions, s.adherences,
                           c.auto_id AS cohort_id,
                           c.cohort_type, c.n_arms,
                           s.sim_type, s.n_iterations
                       FROM simulations AS s, cohorts AS c
                       WHERE c.cohort_type = '{cohort_name}'
                           and c.n_arms = {N}
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
    other_policies_res = out[out.policy_type != "ProbFairPolicy"]
    res = pd.concat([prob_fair_res, other_policies_res])
    
    # Filter intervention effect outside of the query
    res['interv'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['intervention_effect']), dtype=float), axis=1)
    res = res.loc[res.apply(lambda x: all(np.array(x.interv) == intervention_effect), axis=1),:]

    # Convert into numpy arrays (action, adherences, policy)
    res['action_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['actions']), dtype='int64').reshape(x['n_iterations'],x['n_arms'], x['horizon']),axis=1)
    res['adherence_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), dtype='int64').reshape(x['n_iterations'], x['n_arms'],
                                                                        x['horizon'] + 1), axis=1)
    res['policy_arrays'] = res.apply(
        lambda x: np.frombuffer(reader.decompress_packet(x['policy']), dtype='float64').reshape(x['n_arms']) if x['policy'] is not None else None, axis=1)
    
    # Buckets and short names
    
    # Drop Duplicates
    
    # Compute reward
    res['local_rewards'] = res['adherence_arrays'].apply(map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(map_localr_to_R)
    
    # Compute intervention benefit
    
    # Compute wasserstein distance
    
    return res # for debugging

    ## Plots
    
    ## Construct filename and save df; LaTeX table
    # res_filename = f'res_{exp_name}_{cohort_name}_N{N}_T{horizon}.csv'
    # path = os.path.join(results_dir, exp_name, res_filename)
    # save_df_to_csv(out_df, path)
    
    # tex_filename = f'tex_{exp_name}_{cohort_name}_N{N}_T{horizon}.txt'
    # tex_path = os.path.join(results_dir, exp_name, tex_filename)
    # TODO: call csv_to_latex_table here.
    
    return res

def save_df_to_csv(res: pd.DataFrame, save_path:str):
    # filter all bytes and np.array objects:
    mask_inv = (res.applymap(type).isin([bytes, np.ndarray])).any(0)
    df = res[res.columns[~mask_inv]]
    print(f'Dropped columns {set(res.columns) - set(df.columns)}')
    # save:
    df.to_csv(save_path, index=False) 
    return

def csv_to_latex_table(csv_path: str, save_path: str, res_filter: dict = {'synth_cohort_subtype': 'random'},
                       policy_buckets=['prob_fair', 'heuristic', 'comparison', 'baseline'], 
                       metrics=['ib', 'wd'], percent_flag=[True,False],
                       sort_order=['lb', 'ub', 'heuristic_interval_len', 'policy_type'],
                       sort_ascending=[False, True, True, False]):
    
    def write_indented_lines(f,strings: [str]):
        for string in strings:
            f.write('\t' + string + '\n')
            
    def write_policy(f, df_slice, policy_names, end_hline='thick'):
        if end_hline=='thick':
            end = ['\t' + r'\specialrule{1.5pt}{1pt}{1pt}']
        elif end_hline=='thin':
            end = ['\t' + r'\specialrule{1pt}{1pt}{1pt}']
        elif end_hline=='cline':
            end=['\t' + r'\cline{3-4}']
        
        result_lines = list(df_slice['result_line'])
        hlines = ['\t' + r'\cline{3-4}']*(len(result_lines)-1) + end
        if len(result_lines) < len(policy_names):
            result_lines = result_lines + [r'& & & \\']
            hlines = [' ']+end
        elif len(result_lines) > len(policy_names):
            policy_names = policy_names + [r' ']*(len(result_lines) - len(policy_names))
        
        write_indented_lines(f, [policy+param+hline for (policy, param, hline) in zip(policy_names, result_lines, hlines)])
    
    def format_heading(heading: str, percent_flag: bool=False):
        if percent_flag:
            return r'\textsc{' + f'{heading}'+r'} (\%)'
        else:
            return r'\textsc{' + f'{heading}'+r'}'
    
    def format_param(policy_bucket: str,
                     ell: float, nu: float, min_sel_frac:float=None, min_pull_per_pd:float=None,
                     ub:float=None, ip_interval_len:int= None):
        # Based on policy_plus_param_names, but instead returns 'param=val'
        if policy_bucket == "prob_fair":
            if ub==None or ub==1.0:
                return r'$\ell={}$'.format(round(ell,2))
            else:
                return r'$u={}$'.format(round(ub,2))
        elif policy_bucket == "heuristic":
            return r'$\nu={}$'.format(int(nu))
            #return r'H$_{}$'.format(heuristic[:4] if heuristic in ['last', 'random'] else heuristic[:5]) + '\n' + r'$\nu$={} '.format(int(nu))
        elif policy_bucket == "comparison":
            return ""
        elif policy_bucket == "ip" and min_sel_frac == 0.0 and min_pull_per_pd == 0:
            return "baseline"
        elif policy_bucket == "ip":
            return r'$\psi={}$'.format(round(float(min_sel_frac),2)) if min_sel_frac != 0 else \
                   r'$\nu={}$'.format(int(ip_interval_len))
        else:
            return ""
        
    def format_result_line(param_str, e_metrics, sigma_metrics, percent_flag):
        return rf'& {param_str} ' + ''.join([f'& {e_metric*100:.2f} ~({sigma_metric:.2f}) ' if flag else f'& {e_metric:.2f} ~({sigma_metric:.2f}) ' for (e_metric, sigma_metric, flag) in zip(e_metrics, sigma_metrics, percent_flag)]) + r'\\' + '\n'
    
    def generate_policy_name(policy_type, RA_flag=False, heuristic=None):
        if heuristic is not None:
            return [format_heading(f'{heuristic.capitalize()}'), 'heuristic']
        if policy_type=='WhittleIndexPolicy':
            if RA_flag:
                return [format_heading('Risk-Aware'), format_heading('Whittle')]
            else:
                return [format_heading('Threshold'), format_heading('Whittle')]
        else:
            return [format_heading(policy_type[:-6])]
    
    def validate_slice(df_slice):
        # the slice should only include one policy and should not be empty
        policy_list = list(df_slice['policy_type'])
        return len(set(policy_list)) == 1
    
    df = pd.read_csv(csv_path)
    df = df.loc[(df[list(res_filter)] == pd.Series(res_filter)).all(axis=1)]
    df = df.sort_values(sort_order,ascending=sort_ascending)
    df = df.where(df.notnull(), None)
    
    # df[tex_result]= r'& param=val & number (number) & number (number) \\'
    for key in ['lb', 'heuristic_interval_len', 'min_sel_frac', 'min_pull_per_pd', 'ub', 'interval_len']:
        if key not in df:
            df[key] = None
    df['param_entry'] = df.apply(lambda x: format_param(policy_bucket=x['policy_bucket'],
                                                        ell=x['lb'],
                                                        nu=x['heuristic_interval_len'],
                                                        min_sel_frac=x['min_sel_frac'],
                                                        min_pull_per_pd=x['min_pull_per_pd'],
                                                        ub=x['ub'],
                                                        ip_interval_len=x['interval_len']), axis=1)
    
    df['result_line'] = df.apply(lambda x: format_result_line(param_str=x['param_entry'],
                                                              e_metrics=[x[f'e_{metric}'] for metric in metrics],
                                                              sigma_metrics=[x[f'sigma_{metric}'] for metric in metrics], 
                                                              percent_flag=percent_flag), axis=1)
    
    
    with open(save_path, 'w') as f:
        # Begin table
        f.write(r'\begin{table}[]'+'\n')
        f.write(r'\begin{tabular}{|ll V l|l|}'+'\n')
        
        # Header row
        metric_headers = [format_heading(metric.upper(), percent_flag) for (metric, percent_flag) in zip(metrics, percent_flag)]
        header_title = r'Policy & '+''.join([r'& ' + metric_header for metric_header in metric_headers])
        header_lines = [r'\specialrule{1pt}{1pt}{1pt}',
                  header_title + r' \\',
                  r'\specialrule{2.5pt}{1pt}{1pt}']
        write_indented_lines(f, header_lines)
        
        
        # Policies
        for policy_bucket in policy_buckets:
            
            if policy_bucket == 'heuristic':
                for heuristic in ['first', 'last', 'random']:
                    df_slice = df.loc[df['heuristic'] == heuristic]
                    assert(validate_slice(df_slice))
                    policy_names = generate_policy_name(list(df_slice['policy_type'])[0], RA_flag=False, heuristic=heuristic)
                    if heuristic != 'random':
                        write_policy(f, df_slice, policy_names, end_hline = 'thin')
                    else:
                        write_policy(f, df_slice, policy_names, end_hline = 'thick')
                    
            elif policy_bucket == 'comparison':
                for RA_flag in [True, False]:
                    if RA_flag:
                        df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                        df_slice = df_slice.loc[df_slice['local_reward']!='belief_identity']
                        
                    else:
                        df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                        df_slice = df_slice.loc[df_slice['local_reward']=='belief_identity']
                    if not df_slice.empty:
                        policy_names = generate_policy_name(list(df_slice['policy_type'])[0], RA_flag, heuristic=None)
                        write_policy(f, df_slice, policy_names, end_hline = 'thick')
                    
            elif policy_bucket == 'baseline':
                df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                for policy in list(df_slice['policy_type']):
                    df_slice = df.loc[df['policy_type']==policy]
                    assert(validate_slice(df_slice))
                    policy_names = generate_policy_name(policy, RA_flag=False, heuristic=None)
                    write_policy(f, df_slice, policy_names, end_hline = 'cline')
                    
            else: 
                df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                policy_names = generate_policy_name(list(df_slice['policy_type'])[0], RA_flag=False, heuristic=None)
                write_policy(f, df_slice, policy_names, end_hline = 'thick')
            
        write_indented_lines(f, [r'\specialrule{1pt}{1pt}{1pt}']) 
        
        # End table
        f.write(r'\end{tabular}%}'+'\n')
        f.write(r'\caption{CAPTION HERE} \label{tab:LABEL}'+'\n')
        f.write(r'\end{table}'+'\n')
        
        # Extra info
        f.write('\n')
        f.write('-'*65+'\n')
        f.write('Reminder: include the following in the header of the paper:\n')
        f.write('\t'+r'\usepackage{booktabs}'+'\n')
        f.write('\t'+r'\usepackage{siunitx}'+'\n')
        f.write('\t'+r'\sisetup{output-exponent-marker=\ensuremath{\mathrm{e}}}'+'\n')
        f.write('Reminder: include the following before the table:'+'\n')
        f.write('\t'+r'\newcolumntype{V}{!{\vrule width 1pt}}'+'\n')
        
    """
    \begin{tabular}{|ll V l|l|}
\specialrule{1pt}{1pt}{1pt} %\hline
    Policy   &        & \(\textsc{IB}\) (\%) & \(\textsc{HHI}\)                 \\ \specialrule{2.5pt}{1pt}{1pt}
\textsc{ProbFair} & \(\ell=0.19\) & $71.44 ~( 6.01)$ & $0.0115 ~( \num{ 1.16e-04 })$
 \\ \cline{3-4} 
 & \(\ell=0.10\) & $79.87 ~( 6.09)$ & $0.0299 ~( \num{ 2.28e-05 })$
 \\ \cline{3-4} 
 & \(\ell=0.01\) &$91.83 ~( 4.74)$ & $0.0476 ~( \num{ 3.00e-05 })$
 \\ \specialrule{1.5pt}{1pt}{1pt} %\hline
\textsc{First} & \(\nu = 6\) &$68.40 ~( 5.99)$ & $0.0108 ~( \num{ 1.11e-05 })$
 \\ \cline{3-4} 
heuristic & \(\nu = 10\) & $78.02 ~( 5.34)$ & $0.0162 ~( \num{ 7.38e-05 })$
 \\ \cline{3-4} 
  & \(\nu = 100\) & $97.57 ~( 2.80)$ & $0.0371 ~( \num{ 4.14e-04 })$
 \\ \specialrule{1pt}{1pt}{1pt} %\hline
  \textsc{Last} & \(\nu = 6\) & $70.86 ~( 5.85)$ & $0.0113 ~( \num{ 3.85e-05 })$
 \\ \cline{3-4} 
 heuristic & \(\nu = 10\) & $81.05 ~( 4.94)$ & $0.0174 ~( \num{ 1.07e-04 })$
\\ \cline{3-4} 
  & \(\nu = 100\) & $98.46 ~( 2.06)$ & $0.0378 ~( \num{ 4.27e-04 })$
 \\ \specialrule{1pt}{1pt}{1pt} %\hline
  \textsc{Random} & \(\nu = 6\) & $72.00 ~( 5.86)$ & $0.0133 ~( \num{ 1.39e-04 })$
 \\ \cline{3-4} 
 heuristic & \(\nu = 10\) & $82.08 ~( 4.71)$ & $0.0229 ~( \num{ 2.31e-04 })$
 \\ \cline{3-4} 
  & \(\nu = 100\) & $98.01 ~( 2.40)$ & $0.0391 ~( \num{ 4.36e-04 })$

 \\ \specialrule{1.5pt}{1pt}{1pt}  %\specialrule{1pt}{1pt}{1pt} %\hline
 \textsc{Risk-Aware} & \(\lambda =20\) & $82.31 ~( 6.00)$ & $0.0201 ~( \num{ 1.00e-03 })$
 \\
 \textsc{Whittle} & &  &
 \\ \specialrule{1.5pt}{1pt}{1pt} % \specialrule{1pt}{1pt}{1pt}%\hline
\textsc{RoundRobin} & & $68.37 ~( 6.27)$ & $0.0100 ~( \num{ 0.00e+00 })$
 \\ \cline{3-4} 
\textsc{Random} & & $62.80 ~( 6.14)$ & $0.0085 ~( \num{ 8.83e-05 })$
 \\ \specialrule{1pt}{1pt}{1pt} %\hline
\end{tabular}%}
\caption{\(\alpha_\text{intrv}=1.1\) and \(T=180\), on the CPAP dataset.}\label{tab:cpap_110_180}
\end{table} 
    """
    return df


def plot_distribution(df, dep_var='action', kind='bar', split_flag=False):
    # plot adherence or action distribution
    # adherence x-lim: [1,T]
    # action x-lim: [0,T]
    # dep var action or adherence
    # kind bar or line
    
    ## Filter df (if not passed in already filtered, which we may prefer(?))
    # TODO: reduce/filter df
    # e.g. on cohort
    # TODO: construct policy_name (see jupyter notebook)
    n_iters = 50
    horizon = 180
    
    # TODO: order list (policies) and palette
    # order_list = ["Prob Fair \n $\ell$ = 0.19", "Prob Fair \n $\ell$ = 0.1", "Prob Fair \n $\ell$ = 0.01",
    #               'TW\nRisk-Aware', "TW \n baseline"]
    order_list = ['NoActPolicy', 'RandomPolicy', 'WhittleIndexPolicy'] 
        
    prob_fair_colors = plt.get_cmap('Reds')(np.linspace(0.4,0.8,3))
    baseline_colors = plt.get_cmap('Greys')(np.linspace(0.4,0.8,3))
    mate21_color = plt.get_cmap('Oranges')(np.linspace(0.4,0.8,3))[1]
    mate21_color = 'tab:cyan'
    palette = [prob_fair_colors[0], prob_fair_colors[1], prob_fair_colors[2], mate21_color, baseline_colors[2]]
    # palette = np.stack((baseline_colors[0], baseline_colors[1], prob_fair_colors[0], prob_fair_colors[1], prob_fair_colors[2], mate21_color, baseline_colors[2]))
    # return palette
    # palette = ['C0', 'tab:blue', 'tab:purple', 'tab:green', 'tab:red', 'tab:grey', 'tab:pink']
    # palette = [prob_fair_colors[0] if x=="Prob Fair \n $\ell$ = 0.19" else 'red' for x in order_list ]
    # print(palette)
    if horizon == 30:
        order_list = ["Prob Fair \n $\ell$ = 0.19", "Prob Fair \n $\ell$ = 0.01",
              'TW\nRisk-Aware', "TW \n baseline"]

        palette = [prob_fair_colors[0], prob_fair_colors[2], mate21_color, baseline_colors[2]]
    

    df = df.loc[df.policy_name.isin(order_list)]
    df = compute_histogram_vals(df, n_iters, horizon, array_type=dep_var)
    
    
    ## Plot settings
    fontsize = 16
    bin_width = 1
    if split_flag:
        rhs = 49
        lhs = 175
        
        f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1,
                                 sharey=True,figsize=(15,4),gridspec_kw={'width_ratios': [50, 7]})
        
        if kind=='line':
            sns.lineplot(x=f'hist_x_{dep_var}', y=f'hist_y_{dep_var}', hue="policy_name", data=df,palette=palette,hue_order=order_list,ax=ax1,legend=False)
            sns.lineplot(x=f'hist_x_{dep_var}', y=f'hist_y_{dep_var}', hue="policy_name", data=df,palette=palette,hue_order=order_list,ax=ax2,legend=False)
        else:
            sns.barplot(x=f'hist_x_{dep_var}', y=f'hist_y_{dep_var}', hue="policy_name", data=df,palette=palette,hue_order=order_list,ci=None,seed=0, ax=ax1)
            sns.barplot(x=f'hist_x_{dep_var}', y=f'hist_y_{dep_var}', hue="policy_name", data=df,palette=palette,hue_order=order_list,ci=None,seed=0, ax=ax2)
            ax1.set_xlabel("")
            ax2.set_xlabel("")
        
        ax1.set_xlim(-1, rhs)
        if dep_var == 'action':
            ax2.set_xlim(lhs, horizon+1)

        elif dep_var == 'adherence':
            ax2.set_xlim(lhs, horizon+2)
        ymin, ymax = plt.ylim()
        print(ymin, ymax)
        plt.ylim(0,ymax)
        
        print(f'Ratio should be: {rhs+1}:{horizon+2-lhs}')
        
        # print(ticks)
        # ax1.get_yaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        
        
        if dep_var == 'action':
            f.text(0.55, -0.01, r'Distribution of arm pulls', ha='center', fontsize=fontsize)
        elif dep_var == 'adherence':
            f.text(0.55, -0.01, r'Distribution of arm adherences', ha='center', fontsize=fontsize)
            
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        # then create a new legend and put it to the side of the figure (also requires trial and error)
        # ax2.legend(loc='upper right', labels=['Arm 1', 'Arm 2'])
        
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()
        # ax2.yaxis.tick_right()
        ax1.set_ylabel('Count', fontsize=fontsize)
        f.subplots_adjust(wspace=0.01)
        
        # f.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85, hspace=0,wspace=0.01)
        # plt.tight_layout(w_pad=0.01)
    else:
        xmin = 70
        xmin=-1
        if horizon == 30:
            plt.figure(figsize=(10,2))
        else:
            plt.figure(figsize=(20,4))
        # ax = sns.barplot(x=df.policy_name, y=df[dep_var], ci=95, capsize=.2,  seed=bootstrap_seed, order=order_list,
        #             palette=palette)
        if kind=='line':
            ax=sns.lineplot(x=f'hist_x_{dep_var}', y=f'hist_y_{dep_var}', hue="policy_name", data=df,palette=palette,hue_order=order_list,ci=95,seed=0)
        else:
            ax=sns.barplot(x=f'hist_x_{dep_var}', y=f'hist_y_{dep_var}', hue="policy_name", data=df,palette=palette,hue_order=order_list,ci=None,seed=0)
        ax.get_legend().remove()
        # sns.histplot(data=df, x='times_pulled', y='count', hue='policy_name', multiple='stack', palette=palette,binwidth=1,hue_order=order_list,alpha=1)
        # plt.title("Expected Total Intervention Benefit by Policy (" + r'$n$' + " sims=500)" + "\n N = 100, k = 20; T = 30")
        # plt.ylabel("Expected Total Intervention Benefit")
        # plt.xlabel("Policy", fontsize=16)
        # dep_var_label = expand_variable_names(dep_var)
        # plt.ylabel(dep_var_label, fontsize=16)
        if dep_var == 'action':
            ax.set_xlabel('Distribution of arm pulls', fontsize=16)
        elif dep_var == 'adherence':
            ax.set_xlabel('Distribution of arm adherences', fontsize=16)
        # plt.ylabel('Count', fontsize=16)
        ax.set_ylabel('Count')
        
        # plt.tick_params(labelsize=8, bottom=False)
        # ax.set(xticklabels=[])
        if dep_var == 'action':
            ax.set_xlim(xmin,horizon+1)

        elif dep_var == 'adherence':
            ax.set_xlim(xmin,horizon+2)
        ymin,ymax = ax.set_ylim()
        ax.set_ylim(0,ymax)
        
        plt.tight_layout()
    # if cpap_flag:
    #     save_name = f'{dep_var}_histogram_CPAP_INTRV{intrv[0]}_N{N}_T{T[0]}.pdf' 
    # else:
    #     save_name = f'{dep_var}_histogram_N{N}_NFWD{NFWD[0]}_NRVS{NRVS[0]}_NRAND{NRAND[0]}_T{T[0]}.pdf' 
        
    # save_path = os.path.join(save_dir,save_name)
    # if save_flag:
    #     plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()  # TODO: configure saving + save string instead of show

    # print(df.groupby('policy_name').mean())

    return

def main(config):

    reader = DBReader(config['database'])
    reader.cursor.execute('set global max_allowed_packet=67108864')

    policy_name_abbrs = {'NoAct': 'NoAct',
                         'Myopic': 'Myopic',
                         'MathProg': 'IP',
                         'ProbFair': 'PF',
                         'Random': 'Rand',
                         'RoundRobin': 'RR',
                         'WhittleIndex_belief_identity': 'TW',
                         'WhittleIndex_mate21_concave': 'TW-RA'}

    #experiments = list(filter(lambda x: x not in (["general","paths","database"]), config.sections()))
    # funcs = [gen_cpap_df, gen_synthetic_df, gen_ip_df, gen_varyub_df][:1]
    
    # experiments = list(filter(lambda x: x not in (["general","paths","database"]), config.sections()))[2:]
    #funcs = [gen_cpap_df, gen_synthetic_df, gen_ip_df, gen_varyub_df][1:2]
    # funcs = [gen_fairness_vary_policy_df]
    # funcs = [f'gen_{experiment}_df' for experiment in experiments]
    
    # funcs = [gen_vary_cohort_composition_df, gen_cpap_df]
    # funcs = [gen_vary_cohort_composition_df]
    funcs = [gen_fairness_vary_policy_df, gen_vary_cohort_composition_df, gen_cpap_df, gen_no_fairness_vary_policy_df][0:]
    experiments = [f'{func.__name__}'[4:-3] for func in funcs][0:]
    
    print(experiments, funcs)
    for etype, f in zip(experiments, funcs):
        print(etype, f, list(config[etype].items()))
        df = f(reader, policy_name_abbrs=policy_name_abbrs, **config[etype])

    # Close the connection
    reader.cursor.close()
    reader.conn.close()
    
    # Plot
    return df
    
def test(reader, config_section):
    # # generate test df
    # policies = ['NoActPolicy', 'RandomPolicy', 'WhittleIndexPolicy']
    # actions = np.zeros((len(policies), n_iters, n_arms, horizon))
    # adherences = np.zeros((len(policies), n_iters, n_arms, horizon))
    
    # rng = np.random.default_rng(seed=0)
    # pull_arms = rng.choice(n_arms, k, replace=False)
    # actions[2,:,pull_arms,:]=1
    # actions[1,...] = rng.choice(2, (1, n_iters, n_arms, horizon))
    # adherences = rng.choice(2, (len(policies), n_iters, n_arms, horizon))
    
    # df= pd.DataFrame({'policy_type': policies, 
    #                   'action_arrays': [actions[0,...], actions[1,...], actions[2,...]],
    #                   'adherence_arrays': [adherences[0,...], adherences[1,...], adherences[2,...]]})
    
    # # TODO: df[policy_name], see jupyter notebook.
    # df['policy_name'] = df['policy_type']
    
    # # append hist x&y axes (pre-melting)
    # # df = compute_histogram_vals(df, n_iters, horizon)
    
    # # plot
    # plot_distribution(df,dep_var='action', split_flag=True)
    # plot_distribution(df,dep_var='adherence')
    
    # # save plot
    # return df
    
    reader = DBReader(OrderedDict(config['database']))
    reader.cursor.execute('set global max_allowed_packet=67108864')
    
    df = gen_no_fairness_vary_policy_df(reader, **config_section)
    
    reader.cursor.close()
    reader.conn.close()
    
    return df

# def plot_exploration(arrays, lbs, ubs, N=100, k=20):
#     colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
#     plt.figure()
#     for array, lb, ub, color in zip(arrays, lbs, ubs, colors):
#         if array is not None:
#             plt.plot(np.sort(array), label=f'lb={lb}, ub={ub}', color=color)
#             n_lb = (k - N*ub)/(lb-ub)
#             # plt.vlines([n_lb], ymin=0, ymax=np.max(ubs), colors=color, linestyles='--')
#             print(n_lb)
#             comparable_policy = np.zeros((N))
#             comparable_policy[:n_lb] = lb
#             comparable_policy[n_lb:] = ub
#             # plt.plot(np.sort(comparable_policy), linestyle='--', label=f'comparable policy for lb={lb}, ub={ub}', color=color)
#     plt.legend()

if __name__ == "__main__":

    arg_groups = simutils.get_args(argv=None)
    config_filename = "../config/example_analysis.ini"
    level = '..'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(os.path.join(level, config_filename))
    reader = DBReader(OrderedDict({k.upper(): v for k,v in config['database'].items()}))
    reader.cursor.execute('set global max_allowed_packet=67108864')
    
    #df = test(reader, config['vary_cohort_composition'])
    df = main(config)
    
    reader.cursor.close()
    reader.conn.close()
