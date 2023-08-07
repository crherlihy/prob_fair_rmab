from collections import OrderedDict
from itertools import chain
from typing import Callable, Union
from math import sqrt
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_margin_of_error_ci(observed_values: np.array, 
                               n_sim_iterations: int):
    '''
    Compute the margin of error and confidence intervals from observations
    :param observed_values: observed simulation values
    :type observed_values: np.array
    :param n_sim_iterations: number of simulations (iterations)
    :type n_sim_iterations: int
    :return: (margin of error, confidence interval)

    '''
    assert n_sim_iterations == len(observed_values)
    df = len(observed_values) - 1
    t_critical = stats.t.ppf(q=0.975, df=df)  # Get the t-critical value*

    sample_mean = np.mean(observed_values)
    sample_stdev = observed_values.std(ddof=1)  # Get the sample standard deviation
    sigma = sample_stdev / sqrt(n_sim_iterations)  # Standard deviation estimate
    margin_of_error = t_critical * sigma

    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)

    return margin_of_error, confidence_interval


def map_adherences_to_localr(adherences_tensor: np.array, 
                             local_r: Callable = lambda x: x) -> np.array:
    '''
    Maps observed adherence results to reward values using local_r()
    
    :param adherences_tensor: observed adherence results
    :type adherences_tensor: np.array
    :param local_r: local reward function, defaults to lambda x: x
    :type local_r: Callable, optional
    :return: local rewards
    :rtype: np.array

    '''
    return np.apply_along_axis(local_r, -1, adherences_tensor)


def map_localr_to_R(localr_tensor: np.array, 
                    global_R: Callable = np.sum, 
                    avg_over_t: bool = False, 
                    time_dim: int = 2) -> np.array:
    '''
    Maps local reward values to global reward
    
    :param localr_tensor: local reward values
    :type localr_tensor: np.array
    :param global_R: global reward function, defaults to np.sum
    :type global_R: Callable, optional
    :param avg_over_t: whether to average over timesteps, defaults to False
    :type avg_over_t: bool, optional
    :param time_dim: time dimension of localr_tensor, defaults to 2
    :type time_dim: int, optional
    :return: global reward value(s)
    :rtype: np.array

    '''
    if avg_over_t:
        T = localr_tensor.shape[time_dim]
        return np.apply_along_axis(lambda x: x*1/T, 
                                   -1, 
                                   np.apply_over_axes(global_R, 
                                                      localr_tensor, [1, 2]).ravel())
    else:
        return np.apply_over_axes(global_R, localr_tensor, [1, 2]).ravel()


def compute_avg_R(ref_alg_R: np.array, n_iter: int) -> [float]:
    '''
    Compute the average global reward
    
    :param ref_alg_R: global reward values (from many simulation iterations)
    :type ref_alg_R: np.array
    :param n_iter: number of simulation iterations
    :type n_iter: int
    :return: (average global reward, standard deviation, margin of error, confidence intervals)
    :rtype: [float]

    '''
    margin_of_error, ci = compute_margin_of_error_ci(observed_values=ref_alg_R, 
                                                     n_sim_iterations=n_iter)
    return np.mean(ref_alg_R), np.std(ref_alg_R), margin_of_error, ci


def compute_ib(no_act_R: np.array, 
               tw_R: np.array, 
               ref_alg_R: np.array, 
               n_iter: int) -> (float):
    '''
    Compute intervention benefit
    
    :param no_act_R: global reward values from NoAct policy simulation results
    :type no_act_R: np.array
    :param tw_R: global reward values from Thresold Whittle policy simulation results
    :type tw_R: np.array
    :param ref_alg_R: global reward values from reference policy simulation results
    :type ref_alg_R: np.array
    :param n_iter: number of iterations per simulation
    :type n_iter: int
    :return: (average intervention benefit, standard deviation, margin of error, confidence intervals)
    :rtype: (float)

    '''
    x = ref_alg_R - no_act_R
    y = tw_R - no_act_R
    z = [i == j for i, j in zip(x, y)]
    ibs = (ref_alg_R - no_act_R)/(tw_R - no_act_R)
    margin_of_error, ci = compute_margin_of_error_ci(observed_values=ibs, 
                                                     n_sim_iterations=n_iter)
    return np.mean(ibs), np.std(ibs), margin_of_error, ci


def compute_pof(tw_R: np.array, 
                ref_alg_R: np.array, 
                n_iter: int = 100) -> (float):
    '''
    Compute the price of fairness
    
    :param tw_R: global reward values from Thresold Whittle policy simulation results
    :type tw_R: np.array
    :param ref_alg_R: global reward values from reference policy simulation results
    :type ref_alg_R: np.array
    :param n_iter: number of iterations per simulation, defaults to 100
    :type n_iter: int, optional
    :return: (average price of fairness, standard deviation, margin of error, confidence intervals)
    :rtype: (float)

    '''
    pofs = (tw_R - ref_alg_R) / tw_R
    margin_of_error, ci = compute_margin_of_error_ci(pofs, n_iter)
    return np.mean(pofs), np.std(pofs), margin_of_error, ci


def compute_hhi(actions_tensor: np.array, 
                k: int, 
                time_dim: int = 2, 
                take_exp_over_iters: bool = True) -> (float):
    '''
    Compute the Herfindahl-Hirschman Index (HHI)
    
    :param actions_tensor: action results 
    :type actions_tensor: np.array
    :param k: budget
    :type k: int
    :param time_dim: dimension of the array associated with timesteps, defaults to 2
    :type time_dim: int, optional
    :param take_exp_over_iters: whether to take expectation over simulation iterations, defaults to True
    :type take_exp_over_iters: bool, optional
    :return: (average HHI, standard deviation, None, None)
    :rtype: (float)

    '''
    T = actions_tensor.shape[time_dim]
    arm_action_sums = np.sum(actions_tensor, axis=time_dim)
    squared_avg_over_t = np.apply_along_axis(lambda x: (x*(1/(k*T)))**2, 
                                             -1, 
                                             arm_action_sums)

    if take_exp_over_iters:
        return (np.mean(np.sum(squared_avg_over_t,axis=1)).ravel()[0], 
                np.std(np.sum(squared_avg_over_t,axis=1)).ravel()[0], 
                None, 
                None)
    else:
        return (np.sum(squared_avg_over_t,axis=1), None, None, None)


def compute_wasserstein_distance(ref_alg_actions: np.array, 
                                 round_robin_actions: np.array, 
                                 tw_actions: np.array,
                                 take_exp_over_iters: bool = True, 
                                 normalize_counts: bool = False, 
                                 n_iter: int = 100,
                                 normalize_wd_by_tw: bool = False) -> tuple:
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

    def convert_actions_to_pull_count_cdf(sim_actions: np.array, 
                                          normalize: bool = False) -> np.array:
        """
        Helper function, takes in an action tensor with dimensions (n_arms, horizon)
        Counts the number of times each arm was pulled (e.g., over all timesteps, within a single simulation)
        @param sim_actions: action tensor with dimension (n_arms, horizon)
        @param normalize: boolean flag; indicates whether to normalize pull counts; defaults to False
        @return: cumulative distribution of pull counts, of dimension (n_iterations)
        """
        times_pulled = np.sum(sim_actions, axis=-1)
        unique_vals, counts = np.unique(times_pulled, return_counts=True)
        uvc_dict = {k:v for k, v in zip(unique_vals, counts)}

        # Not all possible cumulative values in {0...180} may be represented. We need a count for EACH possible value
        val_counts = {i: 0 if i not in uvc_dict.keys() else int(uvc_dict[i]) for i in range(sim_actions.shape[-1] + 2)}
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
    tw_pull_count_dists = [convert_actions_to_pull_count_cdf(x, normalize_counts) for x in tw_actions]

    ref_wd_vals = [wasserstein(ref_alg, rr) for (ref_alg, rr) in zip(ref_pull_count_dists, rr_pull_count_dists)]

    if normalize_wd_by_tw:
        tw_wd_vals = [wasserstein(tw_alg, rr) for (tw_alg, rr) in zip(tw_pull_count_dists, rr_pull_count_dists)]
        ref_tw_normalized_vals = np.array(ref_wd_vals)/np.array(tw_wd_vals)
        margin_of_error, ci = compute_margin_of_error_ci(np.array(ref_tw_normalized_vals), n_iter)

    else:
        margin_of_error, ci = compute_margin_of_error_ci(np.array(ref_wd_vals), n_iter)

    out = ref_tw_normalized_vals if normalize_wd_by_tw else ref_wd_vals

    if take_exp_over_iters:
        return np.mean(out), np.std(out), margin_of_error, ci
    else:
        return out, None, None, None


def gen_synth_cohort_subtype(row, N: int):
    '''
    Helper function to assign a type to a cohort of arms
    
    :param row: row of a pd.DataFrame
    :param N: number of arms in the cohort
    :type N: int
    :return: descriptor of cohort composition
    :rtype: str

    '''
    if row['n_forward'] == N:
        return 'forward'
    elif row['n_reverse'] == N:
        return 'reverse'
    elif row['n_random'] == N:
        return 'random'
    elif row['n_forward'] + row['n_reverse'] == N:
        return 'mixed'
    else:
        pct_convex = 100*row["pct_convex"]/N
        return "{:.0%}".format(pct_convex)


def map_policy_to_bucket(policy_type: str, sim_type: str = None):
    '''
    Helper function to bucket policies.
    
    :param policy_type: Policy class name
    :type policy_type: str
    :param sim_type: Simulation class name, defaults to None
    :type sim_type: str, optional
    :return: descriptor of policy type
    :rtype: str

    '''
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


def policy_plus_param_names(abbr_dict: dict, 
                            policy_type: str, 
                            policy_bucket: str, 
                            heuristic: str,
                            ell: float, 
                            nu: float, 
                            local_r: str, 
                            min_sel_frac: float = None, 
                            min_pull_per_pd: float = None,
                            ub: float = None, 
                            ip_interval_len: int = None):
    '''
    Helper function to get the policy for result annotations
    
    :param abbr_dict: policy abbreviations dict
    :type abbr_dict: dict
    :param policy_type: policy class name
    :type policy_type: str
    :param policy_bucket: policy bucket (e.g., baseline)
    :type policy_bucket: str
    :param heuristic: heuristic type (e.g. first)
    :type heuristic: str
    :param ell: lower bound (probabilistic fairness)
    :type ell: float
    :param nu: time interval (time-indexed fairness)
    :type nu: float
    :param local_r: local reward function
    :type local_r: str
    :param min_sel_frac: minimum selection fraction, defaults to None
    :type min_sel_frac: float, optional
    :param min_pull_per_pd: minimum number of pulls per period, defaults to None
    :type min_pull_per_pd: float, optional
    :param ub: upper bound (probabilistic fairness), defaults to None
    :type ub: float, optional
    :param ip_interval_len: integer program interval length, defaults to None
    :type ip_interval_len: int, optional
    :return: formatted policy string
    :rtype: str

    '''
    if policy_bucket == "prob_fair":
        return f"{abbr_dict['ProbFair']}\n" + r'$\ell$={}'.format(round(ell, 2))
    elif policy_bucket == "heuristic":
        return r'H$_{}$'.format(heuristic[:1].upper()) + '\n' + r'$\nu$={} '.format(int(nu))
    elif policy_bucket == "comparison":
        return abbr_dict["{}_{}".format(policy_type.replace("Policy", ""), local_r)]
    elif policy_bucket == "ip" and min_sel_frac == 0.0 and min_pull_per_pd == 0:
        return "{}\n".format(abbr_dict[policy_type.replace("Policy", "")]) + "baseline"
    elif policy_bucket == "ip":
        return "{}\n".format(abbr_dict[policy_type.replace("Policy", "")]) + r'$\psi$={} '.format(round(float(min_sel_frac), 2)) if min_sel_frac != 0 else \
               "{}\n".format(abbr_dict[policy_type.replace("Policy", "")])+ r'$\nu$={} '.format(int(ip_interval_len))
    else:
        return abbr_dict[policy_type.replace("Policy", "")]


def compute_expected_num_pulls(policy_bucket: str, 
                               ell: Union[float, None], 
                               nu: Union[float, None], 
                               horizon: int) -> Union[str, float]:
    """
    Compute expected number of pulls based on ell and T (ProbFair) or nu and T (TW heuristics); otherwise, return NaN
    @param policy_bucket: Policy bucket (computed by calling `map_policy_to_bucket`)
    @param ell: Lower bound on probability an arm will be pulled at each timestep for ProbFair; NaN for other policies
    @param nu: Interval length (an integer for TW heuristics, and NaN for other policies)
    @param horizon: policy time horizonn
    @return: Expected number of pulls (for policies whose policy bucket != ProbFair or TW Heuristic, returns np.nan)
    """
    if policy_bucket == "prob_fair":
        return int(float(ell)*int(horizon))
    elif policy_bucket == "heuristic":
        return int(int(horizon)/int(nu))
    else:
        return policy_bucket

def compute_histogram_vals(df: pd.DataFrame, 
                           n_iters: int, 
                           horizon: int, 
                           array_type: str = 'action'):
    '''
    Computes histogram values
    
    :param df: DataFrame of observed values
    :type df: pd.DataFrame
    :param n_iters: number of iterations per simulation
    :type n_iters: int
    :param horizon: simulation horizon
    :type horizon: int
    :param array_type: type of observed values (action or adherence), defaults to 'action'
    :type array_type: str, optional
    :return: cumulated histogram values
    :rtype: pd.DataFrame

    '''

    def get_cumulative_pull_count_freqs(row):
        '''
        Helper function to get cumulative counts per row
        
        :param row: row of pd.DataFrame
        :return: cumulative counts
        :rtype: pd.DataFrame

        '''

        arr = row[f'{array_type}_arrays']
        # times_pulled has shape n_iters*N; contains values for total number of pulls each arm (N=100)
        # received in each simulation iteration (n_iters=100)
        times_pulled = np.sum(arr, axis=-1)

        # Unique vals = number of unique cumulative pull counts a given arm received over the simulation iterations
        # Counts = the number of time an arm received that number of cumulative pulls *over all simulation iterations*
        # unique_vals, counts = np.unique(times_pulled, return_counts=True, axis=-1) # orig. axis needs to be None so
        #   we don't get array counts
        unique_vals, counts = np.unique(times_pulled, return_counts=True)
        uvc_dict = dict(zip(unique_vals, counts))
        
        # Not all possible cumulative values in {0...180} may be represented. We need to get a count for ALL possible values
        val_counts = {i: 0 if i not in uvc_dict.keys() else int(uvc_dict[i]) for i in range(horizon + 2)}

        data = {'vals': list(chain.from_iterable([np.repeat(k, v) for k, v in val_counts.items()])),
                'policy_short_name': row['policy_short_name'], 
                'policy_bucket': row['policy_bucket']}

        temp = pd.DataFrame.from_dict(data)
        return temp

    combined_df = pd.DataFrame(columns=['vals', 'policy_short_name', 'policy_bucket'])

    for i, row in df.iterrows():
        combined_df = combined_df.append(get_cumulative_pull_count_freqs(row))

    combined_df['vals'] = combined_df.vals.astype(int)

    return combined_df


def save_df_to_csv(res: pd.DataFrame, save_path: str):
    '''
    Saves results to csv
    
    :param res: DESCRIPTION
    :type res: pd.DataFrame
    :param save_path: path to save directory
    :type save_path: str
    :return: Saves to csv file
    :rtype: None

    '''
    # filter all bytes and np.array objects:
    mask_inv = (res.applymap(type).isin([bytes, np.ndarray])).any(0)
    df = res[res.columns[~mask_inv]]
    print(f'Dropped columns {set(res.columns) - set(df.columns)}')
    # save:
    df.to_csv(save_path, index=False)
    return


def write_indented_lines(f, strings: [str]):
    '''
    Helper function to write lines with whitespace
    
    :param f: file
    :param strings: iterable of lines to write
    :type strings: [str]
    :return: writes to file
    :rtype: None

    '''
    for string in strings:
        f.write('\t' + string + '\n')


def write_policy(f,  
                 df_slice: pd.DataFrame, 
                 policy_names: (str), 
                 end_hline: str = 'thick', 
                 grp_by: Union[None, str] = None):
    '''
    Helper function to write policies to file
    
    :param f: file
    :param df_slice: slice of a pd.DataFrame
    :type df_slice: pd.DataFrame
    :param policy_names: iterable of policy names
    :type policy_names: (str)
    :param end_hline: formatting of horizontal divider, defaults to 'thick'
    :type end_hline: str, optional
    :param grp_by: whether to group policies, defaults to None
    :type grp_by: Union[None, str], optional
    :return: writes to file
    :rtype: None

    '''
    if end_hline == 'thick':
        end = ['\t' + r'\specialrule{1.5pt}{1pt}{1pt}']
    elif end_hline == 'thin':
        end = ['\t' + r'\specialrule{1pt}{1pt}{1pt}']
    elif end_hline == 'cline':
        end = ['\t' + r'\cline{3-4}']

    if grp_by is not None:
        grp_by_lines = [str(int(df_slice[grp_by]))]

    result_lines = list(df_slice['result_line'])
    hlines = ['\t' + r'\cline{3-4}'] * (len(result_lines) - 1) + end
    if len(result_lines) < len(policy_names):
        result_lines = result_lines + [r'& & & \\']
        hlines = [' '] + end
    elif len(result_lines) > len(policy_names):
        policy_names = policy_names + [r' '] * (len(result_lines) - len(policy_names))

    if grp_by is not None:
        write_indented_lines(f, 
                             [grp + policy + param + hline for (grp, policy, param, hline) in zip(grp_by_lines, policy_names, result_lines, hlines)])
    else:
        write_indented_lines(f, 
                             [policy + param + hline for (policy, param, hline) in zip(policy_names, result_lines, hlines)])


def format_heading(heading: str, percent_flag: bool = False):
    '''
    Helper function formats the heading of a LaTeX table
    
    :param heading: heading description
    :type heading: str
    :param percent_flag: whether to include a percent sign, defaults to False
    :type percent_flag: bool, optional
    :return: Heading, properly formatted for LaTeX
    :rtype: str

    '''
    print(heading)
    heading = "EMD" if heading.lower() == "wd" else heading # call it EMD in the paper to be consistent with eval metrics section
    if percent_flag:
        return r'$\mathbb{E}[\textsc{' + f'{heading}' + r'}]$ (\%)'  # ($\pm$)
    else:
        return r'$\mathbb{E}[\textsc{' + f'{heading}' + r'}]$'  # ($\pm$)


def format_policy_name(policy: str, percent_flag: bool = False):
    '''
    Helper function formats the policy row heading of a LaTeX table
    
    :param policy: policy name
    :type policy: str
    :param percent_flag: whether to include a percent sign, defaults to False
    :type percent_flag: bool, optional
    :return: policy label
    :rtype: str

    '''

    policy = "RR" if policy.lower() == "roundrobin" else "PF" if policy.lower() == "probfair" else policy
    if percent_flag:
        return r'\textsc{' + f'{policy}' + r'} (\%)'
    else:
        return r'\textsc{' + f'{policy}' + r'} '


def format_param(policy_bucket: str,
                 ell: float, 
                 nu: float, 
                 min_sel_frac: float = None, 
                 min_pull_per_pd: float = None,
                 ub: float = None, 
                 ip_interval_len: int = None):
    '''
    Helper function formats parameters for LaTeX tables
    
    :param policy_bucket: bucket of policy
    :type policy_bucket: str
    :param ell: lower bound (probabilistic fairness)
    :type ell: float
    :param nu: time-indexed fairness parameter
    :type nu: float
    :param min_sel_frac: minimum selection parameter, defaults to None
    :type min_sel_frac: float, optional
    :param min_pull_per_pd: minimum number of pulls per period, defaults to None
    :type min_pull_per_pd: float, optional
    :param ub: upper bound (probabilistic fairness), defaults to None
    :type ub: float, optional
    :param ip_interval_len: IP interval length parameter, defaults to None
    :type ip_interval_len: int, optional
    :return: parameter, properly formatted
    :rtype: str

    '''
    # Based on policy_plus_param_names, but instead returns 'param=val'
    if policy_bucket == "prob_fair":
        if ub is None or ub == 1.0:
            return r'$\ell={}$'.format(round(ell, 2))
        else:
            return r'$u={}$'.format(round(ub, 2))
    elif policy_bucket == "heuristic":
        return r'$\nu={}$'.format(int(nu))
        # return r'H$_{}$'.format(heuristic[:4] if heuristic in ['last', 'random'] else heuristic[:5]) + '\n' + r'$\nu$={} '.format(int(nu))
    elif policy_bucket == "comparison":
        return ""
    elif policy_bucket == "ip" and min_sel_frac == 0.0 and min_pull_per_pd == 0:
        return "baseline"
    elif policy_bucket == "ip":
        return r'$\psi={}$'.format(round(float(min_sel_frac), 2)) if min_sel_frac != 0 else \
            r'$\nu={}$'.format(int(ip_interval_len))
    else:
        return ""


def format_result_line(param_str: str, 
                       e_metrics: [float], 
                       sigma_metrics: [float], 
                       percent_flag: [bool]):
    '''
    Helper function formats a result for a LaTeX table
    
    :param param_str: parameter string
    :type param_str: str
    :param e_metrics: expected value metrics
    :type e_metrics: [float]
    :param sigma_metrics: sigma metrics
    :type sigma_metrics: [float]
    :param percent_flag: whether to include percent signs
    :type percent_flag: [bool]
    :return: result line of LaTeX table
    :rtype: str

    '''
    return rf'& {param_str} ' + ''.join(
        [f'& {e_metric * 100:.2f} ~({sigma_metric:.2f}) ' if flag else f'& {e_metric:.2f} ~({sigma_metric:.2f}) '
         for (e_metric, sigma_metric, flag) in zip(e_metrics, sigma_metrics, percent_flag)]) + r'\\' + '\n'


def format_result_line_moe(param_str: str, 
                           e_metrics: [float], 
                           moe_metrics: [float], 
                           percent_flag: [bool], 
                           keep_param_line: bool = True):
    '''
    Helper function formats a result (with margin of error) for a LaTeX table
    
    :param param_str: parameter string
    :type param_str: str
    :param e_metrics: expected value metrics
    :type e_metrics: [float]
    :param moe_metrics: margin of errors
    :type moe_metrics: [float]
    :param percent_flag: whether to include percent signs
    :type percent_flag: [bool]
    :param keep_param_line: whether to include the parameter, defaults to True
    :type keep_param_line: bool, optional
    :return: result line of LaTeX table
    :rtype: str

    '''
    res_line = ''.join(
            [f'& {e_metric * 100:.2f} ~$\pm$ {moe_metric * 100:.2f} ' if flag else f'& {e_metric:.2f} ~$\pm$ {moe_metric:.2f} '
             for (e_metric, moe_metric, flag) in zip(e_metrics, moe_metrics, percent_flag)]) + r'\\' + '\n'

    return rf'& {param_str} ' + res_line if keep_param_line else res_line


def bold_metric_in_result_line(res_line, 
                               metric_idx: int, 
                               n_metrics: int = 2) -> str:
    """
    Helper function to bold the best result for a given row x metric (e.g., bold the row's E[IB] or E[EMD] value)
    @param res_line: Original result line, before values are bolded (generated by calling `format_result_line_moe`
    @param metric_idx: integer id for the metric in question (needs to map to order of metric columns in result table)
    @param n_metrics: total number of metrics (defaults to 2)
    @return: Modified result string
    """
    result_items = res_line.split("&")
    item_to_bold = result_items[metric_idx + 1].strip()

    if metric_idx + 1 < n_metrics:
        items_after_bold_res = ''.join(result_items[metric_idx + 2:]) if metric_idx + 1 < n_metrics else result_items[metric_idx + 2]
        return r'&' + '&'.join(result_items[:metric_idx+1]) + r'\textbf{' + item_to_bold + r'} &' + items_after_bold_res
    else:
        return '&'.join(result_items[:metric_idx+1]) + r' & \textbf{' + item_to_bold.replace(r'\\', r'') + r'} \\'


def generate_policy_name(policy_type: str, 
                         RA_flag: bool = False, 
                         heuristic: str = None):
    '''
    Helper function to generate a policy name
    
    :param policy_type: policy class name
    :type policy_type: TYPE
    :param RA_flag: whether the policy is Risk-Aware Whittle, defaults to False
    :type RA_flag: bool, optional
    :param heuristic: heuristic type, defaults to None
    :type heuristic: str, optional
    :return: formatted policy name
    :rtype: [str]

    '''
    if heuristic is not None:
        return [format_policy_name(f'{heuristic.capitalize()}'), 'heuristic']
    if policy_type == 'WhittleIndexPolicy':
        if RA_flag:
            return [format_policy_name('Risk-Aware'), format_policy_name('Whittle')]
        else:
            return [format_policy_name('Threshold'), format_policy_name('Whittle')]
    else:
        return [format_policy_name(policy_type[:-6])]


def generate_policy_string(policy_type: str, 
                           RA_flag: bool = False, 
                           heuristic: str = None):
    '''
    Helper function to generate a policy string
    
    :param policy_type: policy class name
    :type policy_type: str
    :param RA_flag: whether the policy is Risk-Aware Whittle, defaults to False
    :type RA_flag: bool, optional
    :param heuristic: heuristic type, defaults to None
    :type heuristic: str, optional
    :return: formatted policy string
    :rtype: str

    '''
    if heuristic is not None:
        # hspace = 20 if heuristic.lower() == 'first' else 21 if heuristic.lower() == 'last' else 7
        return r'\textsc{' + f'{heuristic.capitalize()}' + r'}  ' + r'\hfill $\nu$ '
    if policy_type == 'WhittleIndexPolicy':
        if RA_flag:
            return 'RA-TW '
        else:
            return 'TW '
    else:
        if policy_type == 'ProbFairPolicy':
            return format_policy_name(policy_type[:-6]) + r'\hfill $\ell$  '
        return format_policy_name(policy_type[:-6]) + r' '


def validate_slice(df_slice):
    '''
    The pd.DataFrame slice should only include one policy and should not be empty.
    
    :param df_slice: slice of the results
    :type df_slice: pd.DataFrame
    :return: True if valid
    :rtype: bool

    '''
    # the slice should only include one policy and should not be empty
    policy_list = list(df_slice['policy_type'])
    return len(set(policy_list)) == 1

def write_result_sub_group(f, 
                           sub_df: pd.DataFrame, 
                           grp_name: str, 
                           metrics: [str]):
    '''
    Helper function to write a sub-group of results to the file
    
    :param f: file
    :param sub_df: slice of results
    :type sub_df: pd.DataFrame
    :param grp_name: group of results to write
    :type grp_name: str
    :param metrics: which metrics to include
    :type metrics: [str]
    :return: writes to file
    :rtype: None

    '''
    # For Table 1, we want to group by the lower bound on expected number of pulls to compare PF and heuristics
    if grp_name not in ["N/A", "comparison", "baseline"]:
        lb = round(sub_df[sub_df.policy_type=='ProbFairPolicy']['lb'].unique()[0], 3)
        interval_len = int(sub_df[sub_df.policy_bucket == 'heuristic']['heuristic_interval_len'].unique()[0])

        grp_info = r'\makecell{\textsc{' + "{}".format(grp_name) + r'}\\ $\ell= ' + "{}".format(lb)\
                   + r'$ \\ $\nu = ' + "{}".format(interval_len) + r'$}'
    else:
        grp_info = grp_name

    string = r''
    num_rows = sub_df.shape[0]

    best_results = OrderedDict()
    for m in metrics:
        best_results[m] = sub_df['e_{}'.format(m)].argmax() if m == 'ib' else sub_df['e_{}'.format(m)].argmin()

    for i, row in sub_df.reset_index().iterrows():
        policy = generate_policy_string(policy_type=row['policy_type'],
                                        RA_flag=row['local_reward'] != 'belief_identity',
                                        heuristic=row['heuristic'])
        res_line = row['result_line']

        for j, m in enumerate(metrics):
            if best_results[m] == i:
                res_line = bold_metric_in_result_line(res_line, metric_idx=j)

        if i == 0:
            string += '\t' + r'\multirow{' + '{}'.format(num_rows) + r'}{*}{' + r'{}'.format(grp_info) + r'} &' + '{}'.format(policy) + res_line
        else:
            string += '\t' + r' &' + '{}'.format(policy) + res_line

    string += '\t' + r'\specialrule{1.5pt}{1pt}{1pt}'
    f.write(string)
    return


def csv_to_latex_table(csv_path: str, 
                       save_path: str, 
                       res_filter: dict = {'synth_cohort_subtype': 'random'},
                       policy_buckets: [str] = ['prob_fair', 'heuristic', 'comparison', 'baseline'],
                       metrics: [str] = ['ib', 'wd'], 
                       percent_flag: [bool] = [True, False],
                       sort_order: [str] = ['lb', 'ub', 'heuristic_interval_len', 'policy_type'],
                       sort_ascending: [bool] = [False, True, True, False], 
                       size: str = 'tiny'):
    '''
    Load csv of results and save in a LaTeX-formatted text file
    
    :param csv_path: path to the csv
    :type csv_path: str
    :param save_path: path to save the LaTeX table
    :type save_path: str
    :param res_filter: results filter, defaults to {'synth_cohort_subtype': 'random'}
    :type res_filter: dict, optional
    :param policy_buckets: groups of policies, defaults to ['prob_fair', 'heuristic', 'comparison', 'baseline']
    :type policy_buckets: [str], optional
    :param metrics: metrics to include, defaults to ['ib', 'wd']
    :type metrics: [str], optional
    :param percent_flag: whether to include percentage signs, defaults to [True, False]
    :type percent_flag: [bool], optional
    :param sort_order: order to sort, defaults to ['lb', 'ub', 'heuristic_interval_len', 'policy_type']
    :type sort_order: [str], optional
    :param sort_ascending: sort order, defaults to [False, True, True, False]
    :type sort_ascending: [bool], optional
    :param size: LaTeX font size parameter, defaults to 'tiny'
    :type size: str, optional
    :return: DataFrame of results
    :rtype: pd.DataFrame

    '''

    df = pd.read_csv(csv_path)

    if len([res_filter.keys()]):
        df = df.loc[(df[list(res_filter)] == pd.Series(res_filter)).all(axis=1)]
    df = df.sort_values(sort_order, ascending=sort_ascending)
    df = df.where(df.notnull(), None)

    for key in ['e_num_pulls', 'lb', 'heuristic_interval_len', 'min_sel_frac', 'min_pull_per_pd', 'ub', 'interval_len']:
        if key not in df:
            df[key] = None
    df['param_entry'] = df.apply(lambda x: format_param(policy_bucket=x['policy_bucket'],
                                                        ell=x['lb'],
                                                        nu=x['heuristic_interval_len'],
                                                        min_sel_frac=x['min_sel_frac'],
                                                        min_pull_per_pd=x['min_pull_per_pd'],
                                                        ub=x['ub'],
                                                        ip_interval_len=x['interval_len']), 
                                 axis=1)

    df['result_line'] = df.apply(lambda x: format_result_line_moe(param_str=x['param_entry'],
                                                                  e_metrics=[x[f'e_{metric}'] for metric in metrics],
                                                                  moe_metrics=[x[f'moe_{metric}'] for metric in metrics],
                                                                  percent_flag=percent_flag), 
                                 axis=1)

    with open(save_path, 'w') as f:
        # Begin table
        f.write(r'\newcolumntype{V}{!{\vrule width 1pt}}' + '\n')
        f.write(r'\begin{table}[]' + '\n')
        f.write(r'\begin{' + size + r'}' + '\n')
        f.write(r'\begin{center}' + '\n')
        f.write(r'\begin{tabular}{|ll V l|l|}' + '\n')

        # Header row
        metric_headers = [format_heading(metric.upper(), percent_flag) 
                          for (metric, percent_flag) in zip(metrics, percent_flag)]
        header_title = r'Policy & ' + ''.join([r'& ' + metric_header 
                                               for metric_header in metric_headers])
        header_lines = [r'\specialrule{1pt}{1pt}{1pt}',
                        header_title + r' \\',
                        r'\specialrule{2.5pt}{1pt}{1pt}']
        write_indented_lines(f, header_lines)

        # Policies
        for policy_bucket in policy_buckets:

            if policy_bucket == 'heuristic':
                for heuristic in ['first', 'last', 'random']:
                    df_slice = df.loc[df['heuristic'] == heuristic]
                    assert (validate_slice(df_slice))
                    policy_names = generate_policy_name(list(df_slice['policy_type'])[0], 
                                                        RA_flag=False,
                                                        heuristic=heuristic)
                    if heuristic != 'random':
                        write_policy(f, df_slice, policy_names, end_hline='thin')
                    else:
                        write_policy(f, df_slice, policy_names, end_hline='thick')

            elif policy_bucket == 'comparison':
                for RA_flag in [True, False]:
                    if RA_flag:
                        df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                        df_slice = df_slice.loc[df_slice['local_reward'] != 'belief_identity']

                    else:
                        df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                        df_slice = df_slice.loc[df_slice['local_reward'] == 'belief_identity']
                    if not df_slice.empty:
                        policy_names = generate_policy_name(list(df_slice['policy_type'])[0], 
                                                            RA_flag, 
                                                            heuristic=None)
                        write_policy(f, df_slice, policy_names, end_hline='thick')

            elif policy_bucket == 'baseline':
                df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                for policy in list(df_slice['policy_type']):
                    df_slice = df.loc[df['policy_type'] == policy]
                    assert(validate_slice(df_slice))
                    policy_names = generate_policy_name(policy, RA_flag=False, heuristic=None)
                    write_policy(f, df_slice, policy_names, end_hline='cline')

            else:
                df_slice = df.loc[df['policy_bucket'] == policy_bucket]
                policy_names = generate_policy_name(list(df_slice['policy_type'])[0], 
                                                    RA_flag=False, 
                                                    heuristic=None)
                write_policy(f, df_slice, policy_names, end_hline='thick')

        write_indented_lines(f, [r'\specialrule{1pt}{1pt}{1pt}'])

        # End table
        f.write(r'\end{tabular}%}' + '\n')
        f.write(r'\end{center}' + '\n')
        f.write(r'\end{' + size + r'}' + '\n')
        f.write(r'\caption{CAPTION HERE} \label{tab:LABEL}' + '\n')
        f.write(r'\end{table}' + '\n')

        # Extra info
        f.write('\n')
        f.write('-' * 65 + '\n')
        f.write('Reminder: include the following in the header of the paper:\n')
        f.write('\t' + r'\usepackage{booktabs}' + '\n')
        f.write('\t' + r'\usepackage{siunitx}' + '\n')
        f.write('\t' + r'\sisetup{output-exponent-marker=\ensuremath{\mathrm{e}}}' + '\n')
        f.write('Reminder: include the following before the table:' + '\n')
        f.write('\t' + r'\newcolumntype{V}{!{\vrule width 1pt}}' + '\n')
    return df


def get_trend_line_summary_stats_across_cohorts(grps_df: pd.DataFrame, 
                                                grp_by_var: str, 
                                                dep_vars: [str] = ['ib', 'wd']) -> dict:
    """
    Fit a trendline across the average values for each cohort for each dv in dep_vars
    @param grps_df: df w/ results for Exp. 2, grouped by the attribute used to vary cohorts (eg, % convex; %non-adhering)
    @param dep_vars: Dependent variables to fit a linear model for and summarize; defaults to [ib, wd]
    @return: Dict of dicts, of the format: {dep_var: {avg_of_avgs, margin of error, lm slope}}
    """
    temp = grps_df[grps_df.policy_type == "ProbFairPolicy"]
    out = {}

    assert grp_by_var in ["pct_convex", "pct_nonadhering"]

    for dv in dep_vars:
        out[dv] = {}
        print(dv)
        X = np.array(temp[grp_by_var]).reshape(-1, 1)
        y = np.array(temp["e_{}".format(dv)]).reshape(-1, 1)
        lm = LinearRegression().fit(X, y)
        # plt.plot(X, lm.predict(X), color='black')

        out[dv]['avg_of_avgs'] = round(float(np.mean(y))*100, 3)
        out[dv]['moe'] = round(float(compute_margin_of_error_ci(y, len(y))[0])*100, 3)
        out[dv]['coef'] = round(float(lm.coef_[0]), 3)

        print("For DV {}, average of averages is: {}; margin of error is: {}; "
              "slope of line is: {}".format(dv, 
                                            out[dv]['avg_of_avgs'], 
                                            out[dv]['moe'], out[dv]['coef']))

    return out
