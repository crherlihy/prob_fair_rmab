import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from src.Database.read import DBReader
import src.Analysis.compute_metrics as cm


def gen_vary_cohort_composition_df(reader: DBReader,
                                   policy_name_abbrs: dict,
                                   results_dir: str,
                                   plots_dir: str,
                                   exp_name: str = 'vary_cohort_composition_experiments',
                                   cohort_name: str = 'SyntheticCohort',
                                   pct_convex_vals: [float] = [0.1*x for x in range(0, 11)],
                                   n: int = 100,
                                   k: int = 20,
                                   horizon: int = 180,
                                   sim_iterations: int = 100,
                                   lb: float = 0.1,
                                   ub: float = 1.0,
                                   interval_len: int = 10,
                                   make_plots: bool = True,
                                   **kwargs):
    '''
    Generate results DataFrame by querying the database
    
    :param reader: DBReader connector to the database
    :type reader: DBReader
    :param policy_name_abbrs: abbreviations to use for the policy names
    :type policy_name_abbrs: dict
    :param results_dir: path to the results directory
    :type results_dir: str
    :param plots_dir: path to the plots directory
    :type plots_dir: str
    :param exp_name:  name of the experiment (collection of simulations), defaults to 'vary_cohort_composition_experiments'
    :type exp_name: str, optional
    :param cohort_name: name of the Cohort, defaults to 'SyntheticCohort'
    :type cohort_name: str, optional
    :param pct_convex_vals: iterable floats of convex parameter values, defaults to [0.1*x for x in range(0, 11)]
    :type pct_convex_vals: [float], optional
    :param n: number of arms in a cohort, defaults to 100
    :type n: int, optional
    :param k: budget, defaults to 20
    :type k: int, optional
    :param horizon: simulation horizon, defaults to 180
    :type horizon: int, optional
    :param sim_iterations: number of iterations in a simulation, defaults to 100
    :type sim_iterations: int, optional
    :param lb: lower bound (probabilistic fairness), defaults to 0.1
    :type lb: float, optional
    :param ub: upper bound (probabilistic fairness), defaults to 1.0
    :type ub: float, optional
    :param interval_len: length of interval (time-indexed fairness), defaults to 10
    :type interval_len: int, optional
    :param make_plots: whether to make plots, defaults to True
    :type make_plots: bool, optional
    :param **kwargs: unused kwargs
    :return: pd.DataFrame of experiment results
    
    Saves results to csv
    Saves LaTeX-formatted table of results
    Saves plots if make_plots is True

    '''
    N = int(n)
    sim_iterations = int(sim_iterations)

    if not os.path.exists(os.path.join(results_dir, exp_name)):
        os.mkdir(os.path.join(results_dir, exp_name))

    ## Query the db
    query = f""" SELECT sc.sim_id, sc.actions, sc.adherences,
                        p.policy_type, p.policy, p.lb, p.ub,
                        sc.cohort_type, sc.n_arms, sc.n_forward, sc.n_reverse, sc.n_concave, sc.n_convex, sc.n_random, sc.local_reward,
                        sc.sim_type, sc.n_iterations, sc.heuristic, sc.heuristic_interval_len,
                        p.k, p.horizon
                 FROM (SELECT s.auto_id AS sim_id,
                           s.actions, s.adherences, s.heuristic, s.heuristic_interval_len,
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
    other_policies_res = out[out['policy_type'] != "ProbFairPolicy"]
    res = pd.concat([prob_fair_res, other_policies_res])

    # Convert into numpy arrays (action, adherences, policy)
    res['action_arrays'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['actions']), 
                                                             dtype='int64').reshape(x['n_iterations'],
                                                                                    x['n_arms'],
                                                                                    x['horizon']), 
                                     axis=1)
    res['adherence_arrays'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['adherences']), 
                                                                dtype='int64').reshape(x['n_iterations'],
                                                                                       x['n_arms'],
                                                                                       x['horizon'] + 1),
                                        axis=1)
    res['policy_arrays'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['policy']), 
                                                             dtype='float64').reshape(x['n_arms'])
                                     if x['policy'] is not None else None,
                                     axis=1)

    # Compute % concave
    res['pct_convex'] = res.apply(lambda x: x['n_convex']/N, axis=1)

    # Buckets and short names
    res['synth_cohort_subtype'] = res.apply(
        lambda row: cm.gen_synth_cohort_subtype(row[['n_random', 'n_forward', 'n_reverse', 'n_concave', 'n_convex', 'pct_convex']], 
                                                N),
        axis=1)
    res['policy_bucket'] = res.apply(lambda x: cm.map_policy_to_bucket(x['policy_type'], 
                                                                       x['sim_type']), 
                                     axis=1)
    res['policy_short_name'] = res.apply(lambda x: cm.policy_plus_param_names(policy_name_abbrs, 
                                                                              x['policy_type'],
                                                                              x['policy_bucket'], 
                                                                              x['heuristic'], 
                                                                              x['lb'],
                                                                              x['heuristic_interval_len'],
                                                                              x['local_reward'], 
                                                                              ub=x['ub']), 
                                         axis=1)
    # Local and global rewards
    res['local_rewards'] = res['adherence_arrays'].apply(cm.map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(cm.map_localr_to_R)

    # Drop Duplicates
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'lb', 'ub', 'heuristic', 'heuristic_interval_len',
                               'synth_cohort_subtype', 'pct_convex'], 
                              keep="last")

    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'moe_ib', 'ci_ib', 'e_wd', 'sigma_wd', 'moe_wd', 'ci_wd']]))

    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()
    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") 
                                  & (res.sim_type == "Simulation") 
                                  & (res.local_reward == "belief_identity"), :]['action_arrays'].values

    # for cohort_subtype in ['forward', 'reverse', 'random', 'mixed']:
    for cohort_subtype in ["{:.0%}".format(x) for x in pct_convex_vals]:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") 
                                                & (res.sim_type == "Simulation") 
                                                & (res.local_reward == "belief_identity")
                                                & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") 
                                            & (res.sim_type == "Simulation") 
                                            & (res.local_reward == "belief_identity")
                                            & (res.synth_cohort_subtype == cohort_subtype), :]['global_rewards'].values

    # for simtype in res.sim_type.unique():
    for simtype in ['Simulation']:
        # for cohort_subtype in res.synth_cohort_subtype.unique():
        for cohort_subtype in ["{:.0%}".format(x) for x in pct_convex_vals]:
            temp = res[(res.sim_type == simtype) & (res.synth_cohort_subtype == cohort_subtype)]

            tw_actions = temp.loc[(res.policy_type == "WhittleIndexPolicy") 
                                  & (temp.sim_type == "Simulation") 
                                  & (temp.local_reward == "belief_identity") 
                                  & (temp.synth_cohort_subtype == cohort_subtype), :]['action_arrays'].values
            first_action = tw_actions[0]

            avg_R_results = temp['global_rewards'].apply(lambda x: cm.compute_avg_R(ref_alg_R=x, n_iter=sim_iterations))

            ib_results = temp['global_rewards'].apply(lambda x: cm.compute_ib(no_act_R=no_act_R_vals[cohort_subtype][0],
                                                                              tw_R=tw_R_vals[cohort_subtype][0],
                                                                              ref_alg_R=x, 
                                                                              n_iter=sim_iterations))

            wd_results_unnorm = temp.apply(lambda row: cm.compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                                       round_robin_actions=round_robin_actions[0],
                                                                                       tw_actions=first_action,
                                                                                       normalize_wd_by_tw=False,
                                                                                       take_exp_over_iters=True,
                                                                                       n_iter=sim_iterations), 
                                           axis=1)

            wd_results = temp.apply(lambda row: cm.compute_wasserstein_distance(ref_alg_actions=row['action_arrays'],
                                                                                round_robin_actions=round_robin_actions[0],
                                                                                tw_actions=first_action,
                                                                                normalize_wd_by_tw=True,
                                                                                take_exp_over_iters=True,
                                                                                n_iter=sim_iterations), 
                                    axis=1)
            temp['e_ib'] = [x[0] for x in ib_results]
            temp['sigma_ib'] = [x[1] for x in ib_results]
            temp['moe_ib'] = [x[2] for x in ib_results]
            temp['ci_ib'] = [x[3] for x in ib_results]
            temp['e_wd'] = [x[0] for x in wd_results]
            temp['sigma_wd'] = [x[1] for x in wd_results]
            temp['moe_wd'] = [x[2] for x in wd_results]
            temp['ci_wd'] = [x[3] for x in wd_results]
            temp['e_wdun'] = [x[0] for x in wd_results_unnorm]
            temp['sigma_wdun'] = [x[1] for x in wd_results_unnorm]
            temp['moe_wdun'] = [x[2] for x in wd_results_unnorm]
            temp['ci_wdun'] = [x[3] for x in wd_results_unnorm]
            temp['e_R'] = [x[0] for x in avg_R_results]
            temp['sigma_R'] = [x[1] for x in avg_R_results]
            temp['moe_R'] = [x[2] for x in avg_R_results]
            temp['ci_R'] = [x[3] for x in avg_R_results]

            out_df = out_df.append(temp)

    cvx_grps_df = get_cvx_groups_df(out_df)

    cm.get_trend_line_summary_stats_across_cohorts(cvx_grps_df, 'pct_convex', ['ib', 'wd'])

    # make plots
    if make_plots:

        facet_plot_vary_cohort_synth(plots_dir, cvx_grps_df, percent_flag=True)

        for dep_var in ['ib', 'wd']:
            plot_vary_cohort_synth(plots_dir, cvx_grps_df, dep_var, percent_flag=True)

    ## Construct filename and save df; LaTeX table
    res_filename = f'res_{exp_name}_{cohort_name}_N{N}_T{horizon}.csv'
    path = os.path.join(results_dir, exp_name, res_filename)
    cm.save_df_to_csv(out_df, path)
    # vary_cohort_synth_csv_to_latex(out_df, pct_convex_vals)

    return res


def get_cvx_groups_df(res_df: pd.DataFrame) -> pd.DataFrame:
    '''
    
    :param res_df: DataFrame of results
    :type res_df: pd.DataFrame
    :return: DataFrame of results, grouped by convex group composition and policy

    '''
    mask_inv = (res_df.applymap(type).isin([bytes, np.ndarray])).any(0)
    df = res_df[res_df.columns[~mask_inv]]
    cvx_grps = df.groupby(['pct_convex', 'policy_short_name', 'policy_type'],
                          as_index=False).last()
    
    return cvx_grps


def plot_vary_cohort_synth(plots_dir: str, 
                           cvx_grps_df: pd.DataFrame, 
                           dep_var: str, 
                           percent_flag: bool = True) -> None:
    """
    Plot line graph for Experiment 2 (pct_convex on x-axis; E[IB]/ E[EMD] on y-axis; one line per policy)
    
    @param plots_dir: Experiment-specific directory to save plots in (set via config)
    @param cvx_grps_df: DataFrame containing results for Experiment 2, grouped by pct_convex
    @param dep_var: Dependent variable to make a plot for (must be in [ib, emd])
    @param percent_flag: Boolean flag for whether to represent y-values as percentages (defaults to True)
    @return: None (plot is generated and saved in plot directory)
    """

    assert dep_var in ['ib', 'wd']

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    dv_name = "Intervention Benefit" if dep_var == "ib" else "Normalized EMD"
    policies = cvx_grps_df.policy_short_name.unique()
    # palette = dict(zip(policies, sns.color_palette('colorblind', n_colors=len(policies))))
    palette = {'NoAct': '#E69F00', 
               'PF\n$\ell$=0.1': '#009E73', 
               'RR': '#CC79A7', 
               'TW': '#56B4E9'}

    g = sns.lineplot(data=cvx_grps_df, 
                     x="pct_convex", 
                     y="e_{}".format(dep_var), 
                     hue="policy_short_name", 
                     palette=palette,
                     linestyle='--', 
                     linewidth=0.9)

    for policy in policies:
        temp = cvx_grps_df[cvx_grps_df.policy_short_name == policy]
        plt.errorbar(data=temp, 
                     x="pct_convex", 
                     y="e_{}".format(dep_var),
                     yerr=list(temp[f'moe_{dep_var}']),
                     solid_capstyle='projecting', 
                     capsize=3, 
                     label=None, 
                     fmt='none',  
                     color=palette[policy])

    # Looks prettier, but our CIs are too narrow for this to show up
    # for policy in cvx_grps_df.policy_short_name.unique():
    #     temp = cvx_grps_df[cvx_grps_df.policy_short_name == policy]
    #     x = temp.pct_convex
    #     y = temp['e_{}'.format(dep_var)]
    #     moe = temp['moe_{}'.format(dep_var)]
    #     plt.fill_between(x, (y-moe), (y+moe), alpha=0.3)

    plt.xlabel("Cohort composition (% convex arms; N=100)")
    # plt.ylabel("Average {} \n (over 100 simulations)".format(dv_name))
    plt.ylabel("Average {}".format(dv_name))

    if percent_flag:
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    if dep_var == "ib":
        plt.legend(ncol=len(policies), 
                   fontsize='small', 
                   markerscale=2, 
                   loc='upper center', 
                   bbox_to_anchor=(0.5, 1.3),
                   fancybox=True)
    else:
        plt.legend().set_visible(False)

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small') # to right of plot

    figure = plt.gcf()
    figure.set_size_inches(4, 2)

    plt.savefig(os.path.join(plots_dir, 
                             "{}_bar_plot_vary_cohort_synth.png".format(dep_var)),
                bbox_inches='tight')
    plt.close()
    return

def facet_plot_vary_cohort_synth(plots_dir: str, 
                                 cvx_grps_df: pd.DataFrame, 
                                 percent_flag: bool = True) -> None:
    '''
    Generate facet plot
    
    :param plots_dir: path to the plots directory
    :type plots_dir: str
    :param cvx_grps_df: results grouped by cohort type (and policy type)
    :type cvx_grps_df: pd.DataFrame
    :param percent_flag: whether to format with a percent sign, defaults to True
    :type percent_flag: bool, optional
    :return: Saves to the plots_dir directory
    :rtype: None

    '''

    temp = cvx_grps_df[['pct_convex', 'policy_short_name', 'policy_type', 'e_R', 'moe_R', 'e_wdun', 'moe_wdun']]
    policies = cvx_grps_df.policy_short_name.unique()
    palette = {'NoAct': '#E69F00', 
               'PF\n$\ell$=0.1': '#009E73', 
               'RR': '#CC79A7', 
               'TW': '#56B4E9', 
               'RA-TW': 'red'}  
    metric_names = {'ib': "Intervention Benefit", 
                    'wd': "Normalized Earth Mover's Distance", 
                    'R': 'Total Reward',
                    'wdun': "Earth Mover's Distance"}

    # Transform wide df to long; get column to indicate if we're working with E[dep_var] or margin of error, and metric.
    long_df = pd.melt(temp, 
                      id_vars=["pct_convex", "policy_short_name"], 
                      value_vars=['e_R', 'moe_R', 'e_wdun', 'moe_wdun'])
    long_df['value_type'] = long_df.variable.apply(lambda x: "avg_value" if x.startswith("e_") 
                                                   else "margin_of_error")
    long_df['eval_metric'] = long_df.variable.apply(lambda x: metric_names[x.split("_")[1]])

    # First, plot the line graphs for the average values
    avg_vals = long_df[long_df.value_type == "avg_value"]
    moe_vals = long_df[long_df.value_type == "margin_of_error"]

    full_df = pd.merge(avg_vals, 
                       moe_vals, 
                       on=['policy_short_name', 'pct_convex', 'eval_metric'], 
                       indicator=True)
    g = sns.FacetGrid(full_df, 
                      col="eval_metric",  
                      hue="policy_short_name", 
                      palette=palette)
    g.map(plt.errorbar, 
          "pct_convex", 
          "value_x", 
          "value_y", 
          solid_capstyle='projecting', 
          capsize=3,
          label=None, 
          marker='.',
          linewidth=0.9)
    
    g.set_titles(col_template="Expected {col_name}")
    g.set_axis_labels("% strictly convex arms; N=100", "")

    # somehow the best option for only putting labels for the lines and not CIs
    for i, axes in enumerate(g.axes.ravel()):
        if i == 0:
            handles, labels = axes.get_legend_handles_labels()
            g.add_legend(handles=handles[:4], 
                         labels=labels[:4],
                         ncol=len(policies), 
                         fontsize='small', 
                         markerscale=2, 
                         loc='upper center',
                         bbox_to_anchor=(0.4, 1.1),
                         fancybox=True)

        axes.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))

    g.legend.set_title(None)

    if percent_flag:
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.savefig(os.path.join(plots_dir, 
                             "facet_plot_vary_cohort_synth.pdf"),
                bbox_inches='tight')

    plt.show()
    plt.close()

    return
