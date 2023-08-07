import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.Database.read import DBReader
import src.Analysis.compute_metrics as cm


def gen_fairness_vary_policy_df(reader: DBReader,
                                policy_name_abbrs: dict,
                                results_dir: str,
                                plots_dir: str,
                                exp_name: str = 'fairness_vary_policy',
                                cohort_name: str = 'SyntheticCohort',
                                n: int = 100,
                                n_random: int = 100,
                                k: int = 20,
                                horizon: int = 180,
                                sim_iterations: int = 100,
                                make_plots: bool = False,
                                **kwargs):
    '''
    Generate results DataFrame by querying the database
    
    :param reader: DBReader connector to the database
    :type reader: DBReader
    :param policy_name_abbrs: abbreviations to use for the policy names
    :type policy_name_abbrs: dict
    :param results_dir: directory of the results
    :type results_dir: str
    :param plots_dir: directory of the plots
    :type plots_dir: str
    :param exp_name: name of the experiment (collection of simulations), defaults to 'fairness_vary_policy'
    :type exp_name: str, optional
    :param cohort_name: name of the cohort, defaults to 'SyntheticCohort'
    :type cohort_name: str, optional
    :param n: number of arms in the cohort, defaults to 100
    :type n: int, optional
    :param n_random: number of arms that are random type, defaults to 100
    :type n_random: int, optional
    :param k: budget, defaults to 20
    :type k: int, optional
    :param horizon: simulation horizon, defaults to 180
    :type horizon: int, optional
    :param sim_iterations: number of iterations per simulation, defaults to 100
    :type sim_iterations: int, optional
    :param make_plots: flag whether to generate plots, defaults to False
    :type make_plots: bool, optional
    :param **kwargs: unused kwargs
    :return: out_df of experiment results
    
    Saves LaTeX-formatted table of results
    Saves plots if make_plots is True

    '''
    N = int(n)
    sim_iterations = int(sim_iterations)
    res_file_name = "res_{}_".format(exp_name) + "{}_N{}_T{}_K{}.csv".format(cohort_name, 
                                                                             str(N), 
                                                                             str(horizon), 
                                                                             str(k))
    tex_file_name = "tex_{}_".format(exp_name) + "{}_N{}_T{}_K{}.txt".format(cohort_name, 
                                                                             str(N), 
                                                                             str(horizon), 
                                                                             str(k))

    if not os.path.exists(os.path.join(results_dir, exp_name)):
        os.mkdir(os.path.join(results_dir, exp_name))

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

    # Generate cohort subtype, policy bucket, and policy short name strings
    res['synth_cohort_subtype'] = res.apply(
        lambda row: cm.gen_synth_cohort_subtype(row[['n_random', 'n_forward', 'n_reverse']], N), 
        axis=1)
    res['policy_bucket'] = res.apply(lambda x: cm.map_policy_to_bucket(x['policy_type'], 
                                                                       x['sim_type']), 
                                     axis=1)
    res['policy_short_name'] = res.apply(lambda x: cm.policy_plus_param_names(policy_name_abbrs, 
                                                                              x['policy_type'],
                                                                              x['policy_bucket'], 
                                                                              x['heuristic'], x['lb'],
                                                                              x['heuristic_interval_len'],
                                                                              x['local_reward'], 
                                                                              ub=x['ub']), 
                                         axis=1)

    # Compute local and global rewards based on adherence information
    res['local_rewards'] = res['adherence_arrays'].apply(cm.map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(cm.map_localr_to_R)

    # # Keep most recent run for each hyperparam combo
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub', 
                               'heuristic', 'heuristic_interval_len',
                               'synth_cohort_subtype'], 
                              keep="last")

    res['e_num_pulls'] = res.apply(lambda x: cm.compute_expected_num_pulls(policy_bucket=x['policy_bucket'], 
                                                                           ell=x['lb'],
                                                                           nu=x['heuristic_interval_len'], 
                                                                           horizon=horizon),
                                   axis=1)

    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_wd', 'sigma_wd']]))

    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()
    tw_actions = OrderedDict()
    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") & (res.sim_type == "Simulation") & (
            res.local_reward == "belief_identity"), :]['action_arrays'].values

    for cohort_subtype in ['forward', 'reverse', 'random', 'mixed']:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") &
                                                (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                                & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") &
                                            (res.sim_type == "Simulation") & (res.local_reward == "belief_identity")
                                            & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

        tw_actions[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") &
                                  (res.sim_type == "Simulation") & (res.local_reward == "belief_identity") &
                                  (res.synth_cohort_subtype == cohort_subtype), :]['action_arrays'].values

    for simtype in res.sim_type.unique():
        for cohort_subtype in res.synth_cohort_subtype.unique():
            temp = res[(res.sim_type == simtype) & (res.synth_cohort_subtype == cohort_subtype)]

            ib_results = temp['global_rewards'].apply(
                lambda x: cm.compute_ib(no_act_R=no_act_R_vals[cohort_subtype][0], 
                                        tw_R=tw_R_vals[cohort_subtype][0],
                                        ref_alg_R=x, 
                                        n_iter=sim_iterations))

            wd_results = temp.apply(lambda row: cm.compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                             round_robin_actions=round_robin_actions[0],
                                                                             tw_actions=tw_actions[cohort_subtype][0],
                                                                             normalize_wd_by_tw=True,
                                                                             take_exp_over_iters=True), axis=1)
            temp['e_ib'] = [x[0] for x in ib_results]
            temp['sigma_ib'] = [x[1] for x in ib_results]
            temp['moe_ib'] = [x[2] for x in ib_results]
            temp['ci_ib'] = [x[3] for x in ib_results]
            temp['e_wd'] = [x[0] for x in wd_results]
            temp['sigma_wd'] = [x[1] for x in wd_results]
            temp['moe_wd'] = [x[2] for x in wd_results]
            temp['ci_wd'] = [x[3] for x in wd_results]

            out_df = out_df.append(temp)

    if make_plots:
        for dep_var in ["ib", "wd"]:
            barplot_fairness_vary_policy(plots_dir=plots_dir, out_df=out_df, dep_var=dep_var)


    save_path = os.path.join(results_dir, exp_name, res_file_name)
    cm.save_df_to_csv(out_df[out_df.ub.isin([np.nan, 1.0]) & (out_df.lb != 0)], save_path=save_path)
    tex_path = os.path.join(results_dir, exp_name, tex_file_name)
    table_grp_e_num_pulls(out_df, 
                          save_path=tex_path,
                          percent_flag=[True,True],metrics=['ib', 'wd'],size='small')

    return out_df


def barplot_fairness_vary_policy(plots_dir: str, 
                                 out_df: pd.DataFrame, 
                                 dep_var: str) -> None:
    '''
    Generate a barplot
    
    :param plots_dir: path to the plots directory
    :type plots_dir: str
    :param out_df: results DataFrame
    :type out_df: pd.DataFrame
    :param dep_var: dependent variable of analysis, e.g. intervention benefit (ib)
    :type dep_var: str
    :return: Saves a plot to the plots_dir
    :rtype: None

    '''

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Remove any ProbFair results where ub != 1
    temp = out_df[out_df.ub.isin([np.nan, 1.0])]

    no_act = out_df[out_df.policy_type == "NoActPolicy"]
    rr = out_df[out_df.policy_type == "RoundRobinPolicy"]
    ra_tw = out_df[(out_df.policy_type == "WhittleIndexPolicy") 
                   & (out_df.local_reward == "mate21_concave")]
    tw = out_df[(out_df.policy_type == "WhittleIndexPolicy") 
                & (out_df.local_reward == "belief_identity") 
                & (out_df.policy_bucket == "comparison")]

    ## Bar plot
    if dep_var == "ib":
        temp = temp[(temp.policy_type != "NoActPolicy") & (temp.policy_bucket != "comparison")]
        policy_order = ['Rand', 'RR', 'PF\n$\\ell$=0.06', 'H$_F$\n$\\nu$=18 ', 'H$_L$\n$\\nu$=18 ',
                        'H$_R$\n$\\nu$=18 ',
                        'PF\n$\\ell$=0.1', 'H$_F$\n$\\nu$=10 ', 'H$_L$\n$\\nu$=10 ', 'H$_R$\n$\\nu$=10 ',
                        'PF\n$\\ell$=0.17', 'H$_F$\n$\\nu$=6 ', 'H$_L$\n$\\nu$=6 ', 'H$_R$\n$\\nu$=6 ']

        g = sns.barplot(data=temp, 
                        x="policy_short_name", 
                        y="e_{}".format(dep_var), 
                        hue='policy_short_name',
                        dodge=False, 
                        palette='gray', 
                        alpha=0.7,
                        order=policy_order)

        g.axhline(y=rr["e_{}".format(dep_var)].median(), 
                  label='Round Robin {}'.format(dep_var), 
                  color='green')
        g.axhline(y=ra_tw["e_{}".format(dep_var)].median(), 
                  label='Risk-Aware TW {}'.format(dep_var),
                  color='blue')
        g.axhline(y=tw["e_{}".format(dep_var)].median(), 
                  label='TW {}'.format(dep_var),
                  color='red')
        # g.errorbar(data=temp, x="policy_short_name", y="e_{}".format(dep_var), yerr="moe_{}".format(dep_var),
        #              solid_capstyle='projecting', capsize=3, fmt='none')

    elif dep_var == "wd":
        temp = temp[(temp.policy_type != "RoundRobinPolicy") & (temp.policy_bucket != "comparison")]
        policy_order = ['Rand', 'PF\n$\\ell$=0.06', 'H$_F$\n$\\nu$=18 ', 'H$_L$\n$\\nu$=18 ',
                        'H$_R$\n$\\nu$=18 ',
                        'PF\n$\\ell$=0.1', 'H$_F$\n$\\nu$=10 ', 'H$_L$\n$\\nu$=10 ', 'H$_R$\n$\\nu$=10 ',
                        'PF\n$\\ell$=0.17', 'H$_F$\n$\\nu$=6 ', 'H$_L$\n$\\nu$=6 ', 'H$_R$\n$\\nu$=6 ']

        g = sns.barplot(data=temp, 
                        x="policy_short_name", 
                        y="e_{}".format(dep_var), 
                        hue='policy_short_name',
                        dodge=False, 
                        palette='gray', 
                        alpha=0.7,
                        order=policy_order)


        g.axhline(y=no_act["e_{}".format(dep_var)].median(), 
                  label='No Act {}'.format(dep_var),
                  color='orange')
        g.axhline(y=ra_tw["e_{}".format(dep_var)].median(), 
                  label='Risk-Aware TW {}'.format(dep_var),
                  color='blue')
        g.axhline(y=tw["e_{}".format(dep_var)].median(), 
                  label='TW {}'.format(dep_var),
                  color='red')

        # g.errorbar(data=temp, x="policy_short_name", y="e_{}".format(dep_var), yerr="moe_{}".format(dep_var),
        #              solid_capstyle='projecting', capsize=3, fmt='none')

        # plt.errorbar("policy_short_name", "e_{}".format(dep_var), "sigma_{}".format(dep_var),  solid_capstyle='projecting', capsize=3,
        #       color="blue",fmt='none')

        # g.set_ylabels("{}".format(dep_var))
        # g.set_xlabels("")

    # labels = ["some name", "some other name", "horizontal"]
    handles, labels = g.get_legend_handles_labels()

    # Slice list to remove first handle
    g.legend(handles=handles[:3], labels=labels[:3])

    g.figure.savefig(os.path.join(plots_dir, 
                                  f"{dep_var}_bar_plot_fairness_vary_policy.png"),
                     bbox_inches='tight')

    plt.show()
    plt.close()

    return


def table_grp_e_num_pulls(res: pd.DataFrame,  
                          save_path: str,
                          metrics: [str], 
                          grp_by: str = 'e_num_pulls', 
                          percent_flag: [bool] = [True, False], 
                          size='tiny',
                          **kwargs) -> None:
    '''
    Save a table formatted in LaTeX of expected number of pulls
    
    :param res: results DataFrame
    :type res: pd.DataFrame
    :param save_path: path to the save location (including file name)
    :type save_path: str
    :param metrics: iterable of metrics to include
    :type metrics: [str]
    :param grp_by: what to group by, defaults to expected number of pulls
    :type grp_by: str, optional
    :param percent_flag: whether to include "(%)" in the table heading, defaults to [True, False]
    :type percent_flag: [bool], optional
    :param size: LaTeX sizing parameter, defaults to 'tiny'
    :type size: TYPE, optional
    :param **kwargs: unused optional kwargs
    :return: Saves a LaTeX-formatted table of result values
    :rtype: None

    '''

    # call E[num pulls] min E[# pulls]?, put policy to param for heuristics and pf (eg, nu, ell)
    # attach results; deal with rounding and notation for +/- errors.

    # filter all bytes and np.array objects:
    mask_inv = (res.applymap(type).isin([bytes, np.ndarray])).any(0)
    df = res[res.columns[~mask_inv]]

    # We don't want the PF result where lb = 0 because we report that in Experiment 3
    # We DO want to keep the NaNs for baseline and RA-TW/TW
    df = df[df['e_num_pulls'].apply(lambda x: isinstance(x, str) or x > 0)]

    # Group by expected number of pulls to facilitate comparison between PF and heuristics where E[#pulls] match
    # sub_df = df.groupby(['e_num_pulls', 'policy_short_name'], dropna=False).last().reset_index()

    grp_by_str = r'$\min_i\mathbb{E}[\text{\# pulls}]$' if grp_by == "e_num_pulls" else ''

    cols_to_keep = [grp_by, 'policy_short_name']
    for m in metrics:
        cols_to_keep += ['e_{}'.format(m), 'moe_{}'.format(m)]

    for key in ['e_num_pulls', 'lb', 'heuristic_interval_len', 'min_sel_frac', 'min_pull_per_pd', 'ub', 'interval_len']:
        if key not in df:
            df[key] = None

    df['param_entry'] = df.apply(lambda x: cm.format_param(policy_bucket=x['policy_bucket'],
                                                        ell=x['lb'],
                                                        nu=x['heuristic_interval_len'],
                                                        min_sel_frac=x['min_sel_frac'],
                                                        min_pull_per_pd=x['min_pull_per_pd'],
                                                        ub=x['ub'],
                                                        ip_interval_len=x['interval_len']), axis=1)

    df['result_line'] = df.apply(lambda x: cm.format_result_line_moe(param_str=x['param_entry'],
                                                                     e_metrics=[x[f'e_{metric}'] for metric in metrics],
                                                                     moe_metrics=[x[f'moe_{metric}'] for metric in metrics],
                                                                     percent_flag=percent_flag, 
                                                                     keep_param_line=False), 
                                 axis=1)
    with open(save_path, 'w') as f:
        # Begin table
        f.write(r'\newcolumntype{V}{!{\vrule width 1pt}}' + '\n')
        f.write(r'\begin{table}[]' + '\n')
        f.write(r'\begin{' + size + r'}' + '\n')
        f.write(r'\begin{center}' + '\n')
        #f.write(r'\begin{tabular}{|c|ll V l|l|}' + '\n')
        f.write(r'\begin{tabular}{|c|l V l|l|}' + '\n')

        # Header row
        metric_headers = [cm.format_heading(metric.upper(), percent_flag) 
                          for (metric, percent_flag) 
                          in zip(metrics, percent_flag)]

        header_title = grp_by_str + r'& Policy ' + ''.join([r'& ' + metric_header for metric_header in metric_headers])
        header_lines = [r'\specialrule{1pt}{1pt}{1pt}',
                        header_title + r' \\',
                        r'\specialrule{2.5pt}{1pt}{1pt}']
        cm.write_indented_lines(f, header_lines)

        print(df[grp_by].unique())

        for val in sorted([x for x in df[grp_by].unique() if isinstance(x, int)]):
            temp = df[df[grp_by] == val if not pd.isna(val) else pd.isna(df[grp_by])]
            cm.write_result_sub_group(f, 
                                      temp, 
                                      "N/A" if pd.isna(val) else str(int(val)), 
                                      metrics=metrics)

        for val in ["comparison", "baseline"]:
            temp = df[df[grp_by] == val]
            cm.write_result_sub_group(f, temp, val, metrics=metrics)
            # cm.write_result_sub_group(f, temp, "N/A" if pd.isna(val) else str(int(val)), metrics=metrics)

        # End table
        f.write(r'\end{tabular}%}' + '\n')
        f.write(r'\end{center}' + '\n')
        f.write(r'\end{' + size + r'}' + '\n')
        f.write(r'\caption{CAPTION HERE}' + '\n' + '\label{tab:exp1table}' + '\n')
        f.write(r'\end{table}' + '\n')


