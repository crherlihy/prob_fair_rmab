import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

from src.Arms.RestlessArm import RestlessArm
from src.Arms.CollapsingArm import CollapsingArm
from src.Cohorts.SyntheticCohort import SyntheticCohort
from src.Policies.WhittleIndexPolicy import WhittleIndexPolicy

def plot_subsidies(arms: [RestlessArm], 
                   policy: WhittleIndexPolicy, 
                   xlog: bool = False, 
                   save_dir: str = ''):
    '''
    Plot subsidies
    
    :param arms: iterable of Arms
    :type arms: [RestlessArm]
    :param policy: WhittleIndexPolicy to simulate
    :param xlog: whether to log the x axis of the plot, defaults to False
    :type xlog: bool, optional
    :param save_dir: path to save directory, defaults to ''
    :type save_dir: str, optional
    :return: None

    '''
    save_name = 'subsidy_vis.pdf'
    
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(8,5),dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    
    color2 = [plt.get_cmap('Reds')(np.linspace(0.4,0.8,3))[i] for i in range(3)][1]
    color1 = [plt.get_cmap('Blues')(np.linspace(0.4,0.8,3))[i] for i in range(3)][1]
    colors = [color2, color1]
    fontsize=16

    for index, arm in enumerate(arms):
        x = np.arange(len(policy.whittle_index_matrix[index,0,1:]))+1
        if xlog==True:
            # x = np.log(x)
            ax.set_xscale('log')
            plt.xticks([0,1,10,20,30,40,50,60,70,80,90,100,200,300])
            plt.xlabel('Timesteps', fontsize=fontsize)
        else:
            plt.xticks(x)
            plt.xlabel('Timesteps, linear scale')

        plt.ylabel('Whittle index values',fontsize=fontsize)
        plt.plot(x, policy.whittle_index_matrix[index,0,1:], 
                 color=colors[index], linestyle = '--', label=f'Arm {index+1}, s=0')
        plt.plot(x, policy.whittle_index_matrix[index,1,1:], 
                 color=colors[index], label=f'Arm {index+1}, s=1')
        plt.hlines(y=0.2, xmin=1, xmax=x[-1], colors='k', linestyles='--')
        plt.legend()
    
    save_path = os.path.join(save_dir,save_name)
    plt.savefig(save_path)
    return
        
        
def compute_min_nu(policy: WhittleIndexPolicy):
    '''
    Compute the minimum nu required
    
    :param policy: WhittleIndexPolicy to simulate
    :return: nu values (one per arm)
    
    '''
    assert(np.shape(policy.whittle_index_matrix)[0]==2)
    
    nus = np.zeros(2)
    
    mins = np.amin(policy.whittle_index_matrix[:,:,1:], axis=(1,2))
    maxs = np.amax(policy.whittle_index_matrix[:,:,1:], axis=(1,2))
    if maxs[0] < mins[1]:
        # then arm 1 is always pulled, arm 0 never pulled
        nus[:] = [1, policy.horizon+1]
    elif maxs[1] < mins[0]:
        # then arm 0 is always pulled, arm 1 never pulled
        nus[:] = [policy.horizon+1, 1]
    else:
        # they cross, but where? 
        if np.all(policy.whittle_index_matrix[0,:,1] <= np.min(policy.whittle_index_matrix[1,:,1])) \
            or np.all(policy.whittle_index_matrix[1,:,1] <= np.min(policy.whittle_index_matrix[0,:,1])):
            # they don't overlap initially
            
            priority_arm = np.unique(np.argmax(policy.whittle_index_matrix[:,:,1], axis=0))
            assert(len(priority_arm)==1)
            priority_arm = priority_arm[0]
            secondary_arm = 0 if priority_arm == 1 else 1
            
            if np.shape(np.where(policy.whittle_index_matrix[secondary_arm,:,2:]>np.max(policy.whittle_index_matrix[priority_arm,:,1])))[-1]==0:
                # Then secondary arm is never pulled
                nus[priority_arm]=1
                nus[secondary_arm]=policy.horizon+1
            else:
                temp1 = np.min(np.argwhere(policy.whittle_index_matrix[secondary_arm,:,1:]>np.max(policy.whittle_index_matrix[priority_arm,:,1]))[:,0])
                temp2 = np.min(np.argwhere(policy.whittle_index_matrix[secondary_arm,:,1:]>np.max(policy.whittle_index_matrix[priority_arm,:,1]))[:,1])
                pull_secondary = np.max([temp1, temp2]) + 1

                nus[priority_arm] = 2 # needs to wait once for pull secondary
                nus[secondary_arm] = pull_secondary
        else:
            nus[:] = 2
        
    return nus
    
        
def plot_min_nu(save_dir: str, 
                iterations: int = 1_000, 
                bin_width = 1):
    '''
    Plot the minimun nu required
    
    :param save_dir: path to save directory
    :type save_dir: str
    :param iterations: number of iterations, defaults to 1_000
    :type iterations: int, optional
    :param bin_width: plotting tick bin widths, defaults to 1
    :return: None

    '''
    save_name = 'min_nu_vis.pdf'
    nus = np.zeros((0,2))
    no_pull_counter = 0
    fail_counter = 0
    seed = 0
    i = 0
    while i < iterations:
        cohort = SyntheticCohort(seed=seed,
                                 arm_type='CollapsingArm',
                                 n_arms=n_arms,
                                 n_forward=n_forward,
                                 n_reverse=n_reverse,
                                 n_concave=n_concave,
                                 n_convex=n_convex,
                                 n_random=n_random,
                                 initial_state = -1)
        
        policy = WhittleIndexPolicy(horizon=horizon,
                                    arms=cohort.arms,
                                    k=k,
                                    arm_type=cohort.arm_type)
        
        # if non_increasing(policy.whittle_index_matrix[:,:,1:]):
        # if np.all([non_increasing(a.belief_chains[:,1:]) for a in policy.arms]):
        if True:
            min_nus = compute_min_nu(policy=policy)
            nus = np.vstack((nus, min_nus))

            if policy.horizon+1 in min_nus:
                no_pull_counter += 1
            
            i+=1
                
        else:
            fail_counter+=1
        seed += 1
            
    print(f'No pull: {no_pull_counter}/{iterations} = {no_pull_counter*100/(iterations):.2f}%, with {fail_counter} fails ({fail_counter*100/(fail_counter+iterations):.2f}%)')
    
    nus = np.sort(nus, axis=1)
    df = pd.DataFrame(nus)
    df = pd.melt(df)
    
    color2 = [plt.get_cmap('Reds')(np.linspace(0.4,0.8,3))[i] for i in range(3)][1]
    color1 = [plt.get_cmap('Blues')(np.linspace(0.4,0.8,3))[i] for i in range(3)][1]
    
    fontsize=16
    
    # print(dfs)
    # return df
    
    f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True)
    # sns.histplot(data=df, x='value', hue='variable', multiple='stack', palette=[color1, color2], bins='auto', ax=ax1, alpha=0.9)
    # sns.histplot(data=df, x='value', hue='variable', multiple='stack', palette=[color1, color2], bins='auto', ax=ax2, alpha=0.9)
    
    sns.histplot(data=df, x='value', hue='variable', multiple='stack', 
                 palette=[color1, color2], bins=366, ax=ax1, alpha=0.9)
    sns.histplot(data=df, x='value', hue='variable', multiple='stack', 
                 palette=[color1, color2], bins=366, ax=ax2, alpha=0.9)
    
   
    ax1.set_xlim(1, 13)
    ticks = np.arange(2-bin_width/2, 13+bin_width/2, bin_width)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([int(np.floor(i)) for i in ticks])
    ax2.set_xlim(policy.horizon-9, policy.horizon+1)
    ticks = np.arange(policy.horizon-8-bin_width/2, policy.horizon+1+bin_width/2, bin_width)
    
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([int(np.ceil(i)) for i in ticks])
    
    ax2.get_yaxis().set_visible(False)
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    
    f.text(0.55, 0, r'Minimum $\nu_i$ satisfied', ha='center', fontsize=fontsize)
    
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    # then create a new legend and put it to the side of the figure (also requires trial and error)
    ax2.legend(loc='upper right', labels=['Arm 1', 'Arm 2'])
    
    ax1.yaxis.tick_left()
    # ax2.yaxis.tick_right()
    # f.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85, hspace=0,wspace=0.1)
    f.subplots_adjust(wspace=0.05)
    ax1.set_ylabel('Count', fontsize=fontsize)
    
    
    
    save_path = os.path.join(save_dir,save_name)
    plt.savefig(save_path)
    
    return
         
if __name__ == '__main__':
    n_arms = 2
    n_forward = n_arms
    n_reverse = 0
    n_concave = n_forward
    n_convex = n_reverse
    n_random = 0
    
    seed = 3
    
    horizon = 365
    k=2
    
    non_increasing = lambda a : np.all(np.logical_or(a[..., :-1]>=a[..., 1:], np.isclose(a[..., :-1], a[..., 1:])))
    
    cohort = SyntheticCohort(seed=seed,
                             arm_type='CollapsingArm',
                             n_arms=n_arms,
                             n_forward=n_forward,
                             n_reverse=n_reverse,
                             n_concave=n_concave,
                             n_convex=n_convex,
                             n_random=n_random,
                             initial_state = -1)
    
    policy = WhittleIndexPolicy(horizon=horizon,
                                arms=cohort.arms,
                                k=k,
                                arm_type=cohort.arm_type)
    
    plot_subsidies(arms=cohort.arms, policy=policy, xlog=True, save_dir='')
