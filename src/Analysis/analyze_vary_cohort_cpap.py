import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from src.Database.read import DBReader
import src.Analysis.compute_metrics as cm

def gen_cpap_df(reader: DBReader,
                policy_name_abbrs: dict,
                plots_dir: str,
                results_dir: str,
                exp_name: str = 'cpap',
                cohort_name: str = 'CPAPCohort',
                pct_nonadhering_vals: [float] = [0.1*x for x in range(0,11)],
                truncate_nonadhering: bool = True,
                intervention_effect: float = 1.1,
                n: int = 100,
                k: int = 20,
                horizon: int = 180,
                sim_iterations: int = 100,
                lb: float = 0.1,
                ub: float = 1.0,
                sigma: float = 1.0,
                make_plots: bool = True,
                **kwargs):
    '''
    Generate results DataFrame by querying the database
    
    :param reader: DBReader connector to the database
    :type reader: DBReader
    :param policy_name_abbrs: abbreviations to use for the policy names
    :type policy_name_abbrs: dict
    :param plots_dir: path to the plots directory
    :type plots_dir: str
    :param results_dir: path to the results directory
    :type results_dir: str
    :param exp_name: name of the experiment (collection of simulations), defaults to 'cpap'
    :type exp_name: str, optional
    :param cohort_name: name of the Cohort, defaults to 'CPAPCohort'
    :type cohort_name: str, optional
    :param pct_nonadhering_vals: iterable floats of non-adhering parameter values, defaults to [0.1*x for x in range(0,11)]
    :type pct_nonadhering_vals: [float], optional
    :param truncate_nonadhering: whether to truncate the non-adhering formatting, defaults to True
    :type truncate_nonadhering: bool, optional
    :param intervention_effect: intervention effect parameter, defaults to 1.1
    :type intervention_effect: float, optional
    :param n: number of arms in the cohort, defaults to 100
    :type n: int, optional
    :param k: budget, defaults to 20
    :type k: int, optional
    :param horizon: simulation horizon, defaults to 180
    :type horizon: int, optional
    :param sim_iterations: number of iterations per simulation, defaults to 100
    :type sim_iterations: int, optional
    :param lb: lower bound (probabilistic fairness), defaults to 0.1
    :type lb: float, optional
    :param ub: upper bound (probabilistic fairness), defaults to 1.0
    :type ub: float, optional
    :param sigma: sigma parameter, defaults to 1.0
    :type sigma: float, optional
    :param make_plots: whether to generate plots, defaults to True
    :type make_plots: bool, optional
    :param **kwargs: unused kwargs
    :return: pd.DataFrame of experiment results
    
    Saves results to csv
    Saves LaTeX-formatted table of results
    Saves plots if make_plots is True

    '''
    N = int(n)
    sim_iterations = int(sim_iterations)

    ## Query the db
    query = f""" SELECT sc.sim_id, sc.actions, sc.adherences,
                        p.policy_type, p.policy, p.lb, p.ub,
                        sc.cohort_type, sc.n_arms, sc.n_nonadhering, sc.sigma, sc.truncate_nonadhering,
                        sc.sim_type, sc.n_iterations, sc.intervention_effect,
                        p.k, p.horizon, sc.local_reward, sc.heuristic, sc.heuristic_interval_len
                 FROM (SELECT s.auto_id AS sim_id,
                           s.actions, s.adherences, s.heuristic, s.heuristic_interval_len, 
                           c.auto_id AS cohort_id,
                           c.cohort_type, c.n_arms, c.n_nonadhering, c.sigma, c.truncate_nonadhering, 
                           s.sim_type, s.n_iterations,  c.intervention_effect, c.local_reward
                       FROM simulations AS s, cohorts AS c
                       WHERE c.cohort_type = '{cohort_name}'
                           and c.n_arms = {N}
                           and c.sigma = {sigma}
                           and c.truncate_nonadhering = {truncate_nonadhering}
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

    # Filter intervention effect outside of the query
    res['interv'] = res.apply(lambda x: np.frombuffer(reader.decompress_packet(x['intervention_effect']),
                                                      dtype=float),
                              axis=1)
    res = res.loc[res.apply(lambda x: all(np.array(x.interv) == intervention_effect), axis=1), :]

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

    # Compute % non-adhering
    res = res[res.n_nonadhering.apply(lambda x: x is not None)] # previous runs
    res['pct_nonadhering'] = res.apply(lambda x: (x['n_nonadhering'] / N), axis=1)

    # Buckets and short names
    res['policy_bucket'] = res.apply(lambda x: cm.map_policy_to_bucket(x['policy_type'], 
                                                                       x['sim_type']), 
                                     axis=1)

    # print(res.columns)
    res['policy_short_name'] = res.apply(lambda x: cm.policy_plus_param_names(policy_name_abbrs, 
                                                                              x['policy_type'],
                                                                              x['policy_bucket'], 
                                                                              None, 
                                                                              x['lb'], 
                                                                              None,
                                                                              x['local_reward'], 
                                                                              ub=x['ub']), 
                                         axis=1)

    # Compute reward
    res['local_rewards'] = res['adherence_arrays'].apply(cm.map_adherences_to_localr)
    res['global_rewards'] = res['local_rewards'].apply(cm.map_localr_to_R)

    # Drop Duplicates (keep most recent run for each hyperparam combo)
    res = res.drop_duplicates(['policy_short_name', 'local_reward', 'heuristic', 'heuristic_interval_len', 'pct_nonadhering', 'truncate_nonadhering'],
                              keep="last")

    out_df = pd.DataFrame(columns=np.concatenate([res.columns.values, ['e_ib', 'sigma_ib', 'e_wd', 'sigma_wd']]))

    no_act_R_vals = OrderedDict()
    tw_R_vals = OrderedDict()

    round_robin_actions = res.loc[(res.policy_type == "RoundRobinPolicy") 
                                  & (res.sim_type == "Simulation") 
                                  & (res.local_reward == "belief_identity"), :]['action_arrays'].values

    for cohort_subtype in [round(x, 15) for x in pct_nonadhering_vals]:
        no_act_R_vals[cohort_subtype] = res.loc[(res.policy_type == "NoActPolicy") 
                                                & (res.sim_type == "Simulation") 
                                                & (res.local_reward == "belief_identity")
                                                & (res.pct_nonadhering == cohort_subtype), :]['global_rewards'].values

        tw_R_vals[cohort_subtype] = res.loc[(res.policy_type == "WhittleIndexPolicy") 
                                            & (res.sim_type == "Simulation") 
                                            & (res.local_reward == "belief_identity")
                                            & (res.pct_nonadhering == cohort_subtype), :]['global_rewards'].values

    for simtype in ['Simulation']:
        # "{:.0%}".format(x)
        for cohort_subtype in [round(x, 15) for x in pct_nonadhering_vals]:
            print(cohort_subtype)
            temp = res[(res.sim_type == simtype) & (res.pct_nonadhering == cohort_subtype)]
            print(temp.shape, temp.policy_short_name.unique())

            tw_actions = temp.loc[(res.policy_type == "WhittleIndexPolicy") 
                                  & (temp.sim_type == "Simulation") 
                                  & (temp.local_reward == "belief_identity") 
                                  & (temp.pct_nonadhering == cohort_subtype), :]['action_arrays'].values
            first_action = tw_actions[0]
            
            
            avg_R_results = temp['global_rewards'].apply(lambda x: cm.compute_avg_R(ref_alg_R=x, 
                                                                                    n_iter=sim_iterations))

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

    cpap_grps_df = get_cpap_groups_df(out_df)

    cm.get_trend_line_summary_stats_across_cohorts(cpap_grps_df, 
                                                   'pct_nonadhering', 
                                                   ['ib', 'wd'])

    # make plots
    if make_plots:

        facet_plot_vary_cohort_cpap(plots_dir, cpap_grps_df, percent_flag=True)

        for dep_var in ['ib', 'wd']:
            plot_vary_cohort_cpap(plots_dir, cpap_grps_df, dep_var, percent_flag=True)


    ## Construct filename and save df; LaTeX table
    res_filename = f'res_{exp_name}_{cohort_name}_N{N}_T{horizon}.csv'
    path = os.path.join(results_dir, exp_name, res_filename)
    cm.save_df_to_csv(out_df, path)

    return res


def get_cpap_groups_df(res_df: pd.DataFrame) -> pd.DataFrame:
    '''
    
    :param res_df: DataFrame of results
    :type res_df: pd.DataFrame
    :return: DataFrame of results, grouped by CPAP cohort type and policy

    '''
    print(res_df.shape)
    mask_inv = (res_df.applymap(type).isin([bytes, np.ndarray])).any(0)
    df = res_df[res_df.columns[~mask_inv]]
    cpap_grps = df.groupby(['pct_nonadhering', 'policy_short_name', 'policy_type'],
                           as_index=False).last()
    print(cpap_grps)
    return cpap_grps


def plot_vary_cohort_cpap(plots_dir: str, 
                          cpap_grps_df: pd.DataFrame, 
                          dep_var: str, 
                          percent_flag: bool = True) -> None:
    """
    Plot line graph for Experiment 2 (pct_nonadhering on x-axis; E[IB]/ E[EMD] on y-axis; one line per policy)
    
    @param plots_dir: Experiment-specific directory to save plots in (set via config)
    @param cpap_grps_df: DataFrame containing results for Experiment 2, grouped by pct_nonadhering
    @param dep_var: Dependent variable to make a plot for (must be in [ib, emd])
    @param percent_flag: Boolean flag for whether to represent y-values as percentages (defaults to True)
    @return: None (plot is generated and saved in plot directory)
    """

    assert dep_var in ['ib', 'wd']

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    cpap_grps_df = cpap_grps_df[cpap_grps_df.policy_short_name.apply(lambda x: x not in ["TW-RA", "Rand"])]

    dv_name = "Intervention Benefit" if dep_var == "ib" else "Normalized EMD"
    policies = cpap_grps_df.policy_short_name.unique()
    # palette = dict(zip(policies, sns.color_palette('colorblind', n_colors=len(policies))))
    palette = {'NoAct': '#E69F00', 
               'PF\n$\ell$=0.1': '#009E73', 
               'RR': '#CC79A7', 
               'TW': '#56B4E9'}

    g = sns.lineplot(data=cpap_grps_df, 
                     x="pct_nonadhering", y="e_{}".format(dep_var), 
                     hue="policy_short_name", 
                     palette=palette,
                     linestyle='--', 
                     linewidth=0.9)

    for policy in policies:
        temp = cpap_grps_df[cpap_grps_df.policy_short_name == policy]
        plt.errorbar(data=temp, 
                     x="pct_nonadhering", 
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
    plt.ylabel("Average {}".format(dv_name))

    # if percent_flag:
        # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

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
                             "{}_bar_plot_vary_cohort_cpap.png".format(dep_var)),
                bbox_inches='tight')
    plt.close()
    return


def facet_plot_vary_cohort_cpap(plots_dir: str, 
                                cpap_grps_df: pd.DataFrame, 
                                percent_flag: bool = True) -> None:
    '''
    Generate a facet plot from the results
    
    :param plots_dir: directory to save the plot
    :type plots_dir: str
    :param cpap_grps_df: results grouped by cohort type (and policy type)
    :type cpap_grps_df: pd.DataFrame
    :param percent_flag: whether to format with a percent sign, defaults to True
    :type percent_flag: bool, optional
    :return: Saves to the plots_dir directory
    :rtype: None

    '''
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    cpap_grps_df = cpap_grps_df[cpap_grps_df.policy_short_name.apply(lambda x: x not in ["TW-RA", "Rand"])]

    temp = cpap_grps_df[['pct_nonadhering', 'policy_short_name', 'policy_type', 'e_R', 'moe_R', 'e_wdun', 'moe_wdun']]
    policies = cpap_grps_df.policy_short_name.unique()
    palette = {'NoAct': '#E69F00', 
               'PF\n$\ell$=0.1': '#009E73', 
               'RR': '#CC79A7', 
               'TW': '#56B4E9'}
    metric_names = {'ib': "Intervention Benefit", 
                    'wd': "Normalized Earth Mover's Distance", 
                    'R': 'Total Reward',
                    'wdun': "Earth Mover's Distance"}

    # Transform wide df to long; get column to indicate if we're working with E[dep_var] or margin of error, and metric.
    long_df = pd.melt(temp, 
                      id_vars=["pct_nonadhering", "policy_short_name"],
                      value_vars=['e_R', 'moe_R', 'e_wdun', 'moe_wdun'])
    long_df['value_type'] = long_df.variable.apply(
        lambda x: "avg_value" if x.startswith("e_") else "margin_of_error")
    long_df['eval_metric'] = long_df.variable.apply(lambda x: metric_names[x.split("_")[1]])

    # First, plot the line graphs for the average values
    avg_vals = long_df[long_df.value_type == "avg_value"]
    emd_avgs = long_df[(long_df.value_type == "avg_value") 
                       & (long_df.eval_metric == "Earth Mover's Distance")]

    moe_vals = long_df[long_df.value_type == "margin_of_error"]

    full_df = pd.merge(avg_vals, 
                       moe_vals, 
                       on=['policy_short_name', 'pct_nonadhering', 'eval_metric'], 
                       indicator=True)
    g = sns.FacetGrid(full_df, 
                      col="eval_metric", 
                      hue="policy_short_name", 
                      palette=palette, 
                      sharey=True)
    g.map(plt.errorbar, 
          "pct_nonadhering", 
          "value_x", 
          "value_y", 
          solid_capstyle='projecting', 
          capsize=3,
          label=None, 
          marker='.', 
          linewidth=0.9)

    g.set_titles(col_template="Expected {col_name}")
    g.set_axis_labels("% non-adhering arms; N=100", "")

    for i, axes in enumerate(g.axes.ravel()):
        # if i == 0:
        #     handles, labels = axes.get_legend_handles_labels()
        #     g.add_legend(handles=handles[:4], labels=labels[:4],
        #                   ncol=len(policies), fontsize='small', markerscale=2, loc='upper center',
        #                   bbox_to_anchor=(0.4, 1.1),
        #                   fancybox=True)
    
        axes.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))

    # g.legend.set_title(None)
    # plt.legend().set_visible(False)

    if percent_flag:
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.savefig(os.path.join(plots_dir, "facet_plot_vary_cohort_cpap.pdf"),
                bbox_inches='tight', )

    plt.show()
    plt.close()

    return
