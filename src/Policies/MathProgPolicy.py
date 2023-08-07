'''MathProgPolicy: Solve an integer program (IP) for the optimal policy 
    given constraints budget (k) and minimum pulls per arm per interval
'''
import warnings
import logging
from operator import ge, eq
from functools import reduce
from collections import OrderedDict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# to pip install gurobipy: python -m pip install -i https://pypi.gurobi.com gurobipy
from gurobipy import GRB
import gurobipy as gp

from src.Policies.Policy import Policy
from src.Arms.RestlessArm import RestlessArm

class MathProgPolicy(Policy):
    '''MathProgPolicy: Solve an integer program (IP) for the optimal policy 
        given constraints budget (k) and minimum pulls per arm per interval
    '''
    def __init__(self, 
                 horizon: int,
                 arms: [RestlessArm],
                 k: int,
                 n_actions: int = 2,
                 pull_action: int = 1,
                 interval_len: int = None,
                 min_sel_frac: float = 0,
                 min_pull_per_pd: int = 0, 
                 type: str = "linear",
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False,
                 **kwargs):
        '''
        
        :param horizon: simulation horizon
        :type horizon: int
        :param arms: group of arms in simulation
        :type arms: [RestlessArm]
        :param k: budget
        :type k: int
        :param n_actions: number of actions, defaults to 2
        :type n_actions: int, optional
        :param pull_action: value of "pull" action, defaults to 1
        :type pull_action: int, optional
        :param interval_len: length of an interval, defaults to None
        :type interval_len: int, optional
        :param min_sel_frac: minimum selection fraction per arm, defaults to 0
        :type min_sel_frac: float, optional
        :param min_pull_per_pd: minimum pulls per period, defaults to 0
        :type min_pull_per_pd: int, optional
        :param type: either linear or integer, defaults to "linear"
        :type type: str, optional              
        :param error_log: error logger, defaults to logging.getLogger('error_log')              
        :type error_log: logging.Logger, optional
        :param verbose: whether to print to the console, defaults to False
        :type verbose: bool, optional
        :param **kwargs: optional unused kwargs
        :return: None

        '''
        Policy.__init__(self, 
                        horizon=horizon,
                        arms=arms,
                        k=k, 
                        error_log=error_log,
                        verbose=verbose, 
                        **kwargs)
        
        self.n_actions = n_actions
        self.pull_action = pull_action
        if interval_len is None:
            self.interval_len = horizon
        else:
            self.interval_len = interval_len
        self.min_sel_frac = min_sel_frac
        self.min_pull_per_pd = min_pull_per_pd
        self.type = type
        
        self.policy = self.solve_mip() # shape: NxT
        

    def compute_game_tree(self, 
                          arm: RestlessArm, 
                          policy: [int], 
                          t_index: int, 
                          g: nx.DiGraph):
        """
        Computes the game tree of a policy, for small number of arms, horizon, and budget.
        
        :param arm: Restless arm to compute game tree for.
        :param policy: Proposed policy. Represents the sequence of actions that we'll 
            use to index into the arm's transition matrix and derive the transition sequence.
        :param t_index: timestep used to index into the policy
        :param g: networkX DiGraph (directed graph)
        :return: networkX DiGraph
        """
        
        # This is our recursive base case; return G when we've completed the sequence of actions prescribed by policy
        # and expanded the final set of transitions for nodes that were leaf nodes at T-1
        if t_index == len(policy) or len(g.nodes()) == len(policy) ** 2 - 1:
            return g

        else:
            leaf_nodes = [node for node in g.nodes() 
                          if (g.out_degree[node] == 0 and g.in_degree[node] == 1)
                          or (node == 0 and len(g.nodes()) == 1)]

            # Get the parent of each leaf node, and also create the leaf node's |Actions| child nodes
            for leaf in leaf_nodes:

                parent_id = list(nx.predecessor(g, leaf).keys())[0]
                child_ids = [parent_id * 2 + 1, parent_id * 2 + 2]

                if self.verbose:
                    print('parent', parent_id, 'leaf', leaf, 'children', child_ids)

                # At each timestep t, we take the action currently prescribed by the policy
                # Given the state we are in at t, and the action we take at t, we transition to s'
                # the probability with which any (s, a, s') transition occurs is dictated by the arm's transition matrix
                for state in range(0, arm.n_states):
                    action_taken = policy[t_index]
                    g.add_node(child_ids[state], data={'state': state})
                    g.add_weighted_edges_from([(parent_id, child_ids[state],
                                                arm.transition[action_taken,
                                                               g.nodes()[parent_id]['data']['state'],
                                                               state])])

            # After we've created a child node corresponding to each possible s' in States,
            # recurse (move one to the right in the policy's action sequence)
            t_index += 1
            return self.compute_game_tree(arm, policy, t_index, g)

    def compute_game_tree_all_actions(self, arm: RestlessArm, t_index: int, g: nx.DiGraph):
        """
        In contrast to compute_game_tree(), which takes the action of the policy as the action taken at each timestep,
        this game tree computes paths for ALL possible action-transition pairs. (I.e., a branching factor of 4 instead of 2)
        :param arm: Restless arm to compute game tree for.
        :param t_index: timestep used to index into the policy
        :param g: networkX DiGraph (directed graph)
        :return:
        """
        actions = list(range(self.n_actions))
        branch_factor = arm.n_states * self.n_actions

        # This is our recursive base case; return G when we've completed a length-T sequence of possible actions
        if t_index == self.horizon or len(g.nodes()) == self.horizon ** branch_factor + branch_factor:
            return g

        else:
            leaf_nodes = [node for node in g.nodes() if (g.out_degree[node] == 0 and g.in_degree[node] == 1)
                          or (node == 0 and len(g.nodes()) == 1)]

            # Get the parent of each leaf node, and also create the leaf node's |Actions| child nodes
            for leaf in leaf_nodes:

                parent_id = list(nx.predecessor(g, leaf).keys())[0]
                child_ids = [(parent_id * branch_factor) + i for i in range(1, branch_factor + 1)]
                counter = 0

                # At each timestep t, we evaluate the probability of each possible state-action -> s' pair
                # Given the state we are in at t, and the action we take at t, we transition to s'
                # the probability with which any (s, a, s') transition occurs is dictated by the arm's transition matrix
                for state in range(0, arm.n_states):
                    for action in actions:
                        g.add_node(child_ids[counter], data={'state': state})
                        
                        g.add_edge(parent_id, child_ids[counter], 
                                   data={'w': arm.transition[action,
                                                             g.nodes()[parent_id]['data']['state'], 
                                                             state],
                                         'act': action})
                        counter += 1

            # After we've created a child node corresponding to each possible s' in States,
            # recurse (increment by one time step)
            t_index += 1
            return self.compute_game_tree_all_actions(arm, t_index, g)

    @staticmethod
    def compute_weighted_path(graph: nx.DiGraph, src_id: int, dst_id: int):
        """
        Helper function to compute the probability of reaching the state implied by a node at time t
        :param graph: the arm's game tree graph (parameterized by a specific policy)
        :param src_id: The root id (0)
        :param dst_id: Target node id
        :return: The path (as a sanity check; it's a binary tree, only 1 path from src -> dst possible); the product of the path's edges
        """
        path = nx.dijkstra_path(graph, src_id, dst_id, weight='w')
        product = reduce((lambda x, y: x * y), [graph.edges()[e]['data']['w'] for e in list(zip(path, path[1:]))])
        return path, product

    def compute_expected_reward(self, graph: nx.DiGraph, depth: int, action_at_t: int, goal_state: int = 1, ):
        """
        Compute the expected reward at time t, where expectation is computed over states
        E[r_t] = \sum_{s \in S} p(s)p(s'|s, a)
        :param graph: the arm's game tree graph (parameterized by a specific policy)
        :param depth: level in the tree to consider (this corresponds to nodes that are leaf nodes at time t)
        :param goal_state: the state associated with >0 reward. Note this implicitly bakes in r(s_{i,t}) = s_{i,t}.
        :return:
        """
        preds = {node: list(graph.predecessors(node))[0] for node in graph.nodes() if node != 0}

        # Get all nodes at this level in the tree if the "state" they represent = goal_state (1)
        dst_nodes = [node for node in list(nx.descendants_at_distance(graph, source=0, distance=depth))
                     if (graph.nodes()[node]['data']['state'] == goal_state) and
                     (graph.edges()[(preds[node], node)]['data']['act'] == action_at_t)]

        # sum over the expected reward represented by each node at level <depth> that represents a path ending in 1 at t
        if self.verbose:
            print([self.compute_weighted_path(graph, 0, dst)[1] for dst in dst_nodes])
            print(np.sum([self.compute_weighted_path(graph, 0, dst)[1] for dst in dst_nodes]))
        expected_reward = np.sum([self.compute_weighted_path(graph, 0, dst)[1] for dst in dst_nodes])
        return expected_reward

    def plot_game_tree(self, g: nx.DiGraph, fig_path: str = None, fig_save: bool = False):
        """
        Helper function to plot the game tree. Nodes are color-coded by what state they represent
        state = 0 nodes are red; state =1 nodes are green
        :param g: game tree graph
        :param fig_path: path to use when saving the figure
        :param fig_save: boolean to tell us whether to save the figure or not
        :return: displays plot
        """

        node_colors = ['green' if g.nodes()[node]['data']['state'] == 1 else 'red' for node in g.nodes()]

        plt.figure(1, figsize=(12, 12))
        pos = nx.planar_layout(g, scale=10)
        nx.draw_networkx_labels(g, pos=pos)
        nx.draw_networkx_edges(g, pos=pos)
        nx.draw_networkx_nodes(g, pos=pos, node_color=node_colors)
        nx.draw_networkx_edge_labels(g, pos=pos)
        plt.show()
        if fig_save and fig_path is not None:
            plt.savefig(fig_path)
        return

    def generate_A_b_single_arm(self, arm: RestlessArm):

        def gen_non_neg_constraints():
            """
            Each arm-timestep-action decision variable must be >= 0
            :param horizon: total number of timesteps
            :return: operator (>=), A, b
            """
            dim = self.horizon * self.n_actions
            op = ge
            A = np.identity(dim)
            b = np.zeros(dim).reshape(dim, 1)
            return op, A, b

        def gen_exactly_one_action_per_t_constraints():
            """
            We must take exactly one action per arm-timestep set of decision variables
            This means that exactly one of the variables associated with a timestep must = 1
            :param horizon: total number of timesteps
            :return: operator (=), A, b
            """

            dim = self.horizon * self.n_actions
            op = eq
            A = np.zeros([self.horizon, dim])
            counter = 0

            # Each arm has a set of decision variables for each timestep; cardinality of this set is |actions|
            # Each decision variable in this set represents whether an action is chosen \in {0,1}
            # We can't just use one decision variable because we need to be able to unroll the objective function
            # We must take exactly one action at each timestep, so these constraints are strict equalities
            for i in range(self.horizon):
                A[i, counter:counter + self.n_actions] = 1
                counter += self.n_actions
            b = np.ones(self.horizon).reshape(self.horizon, 1)

            return op, A, b

        def gen_min_selection_fraction():
            """
            We want the fraction of times that an arm is pulled (over all timesteps) to be at least the min_selection_fraction
            :param min_selection_fraction: minimum selection fraction in (0,1)
            :param horizon: total number of timesteps
            :param pull_action: within the action vector, the index corresponding to "pull" (for our case, it's 1, but trying to make generalizing possible here)
            :return: operator (>=), A, b
            """

            dim = self.horizon * self.n_actions
            op = ge

            # We want to ensure that the arm is pulled at least min_selection_fraction % of the time
            # For each arm-timestep-|actions| set of decision variables, the <pull_action>th one corresponds to action_t = pull
            # We need to sum over this subset of relevant decision variables for all timesteps and make that sum >= min_sel_frac*T
            A = np.zeros(dim).reshape(1, dim)
            A[0, [x + self.pull_action for x in range(0, dim, self.n_actions)]] = 1
            b = np.array([self.min_sel_frac * self.horizon]).reshape(1, 1)

            return op, A, b

        non_neg_constraints = gen_non_neg_constraints()
        exactly_one_action_per_t_constraints = gen_exactly_one_action_per_t_constraints()
        min_sel_frac_constraints = gen_min_selection_fraction()

        return non_neg_constraints, exactly_one_action_per_t_constraints, min_sel_frac_constraints

    def compute_obj_function_coefficients(self, arm: RestlessArm):
        '''
        Compute objective function coefficients of an arm
        
        :param arm: arm in cohort being optimized
        :type arm: RestlessArm
        :return: (horizon*n_actions, )-length array of coefficents
        :rtype: np.array

        '''
        n_coeffs = self.horizon * self.n_actions
        c = np.zeros([n_coeffs])

        G = nx.DiGraph()
        G.edges(data=True)
        G.add_node(0, data={'state': 1})

        G = self.compute_game_tree_all_actions(arm, t_index=0, g=G)
        #
        # if self.verbose:
        #     self.plot_game_tree(G)

        counter = 0
        actions = list(range(self.n_actions))
        for t in range(0, self.horizon):
            for action in actions:
                denom = 2 ** t
                c[counter] = self.compute_expected_reward(graph=G, depth=t + 1, goal_state=1,
                                                          action_at_t=action) / denom
                counter += 1
        return c

    def compute_interval_lookup(self):
        """
        Helper function that computes ceil(horizon/interval_length) intervals, and then creates a look-up dictionary
        with integer timesteps in [0,horizon] as keys and the interval associated with each timestep as values.
        Example: for horizon = 10 and interval_length = 2, there are five intervals containing 2*n_actions timesteps each
        lookup[0] = 0; lookup[1] = 0; lookup[2] = 0; lookup[3] = 0; lookup[4] = 1 ...
        :return: intervals (a range with interval_length step size), interval lookup dictionary
        """
        
        interval_dict = OrderedDict()

        offset = self.n_actions * (self.interval_len - (self.horizon % self.interval_len))
        intervals = list(range(0, (self.horizon * self.n_actions + 1) + offset, self.interval_len * self.n_actions))

        for i, r in enumerate(intervals[:-1]):
            for timestep in list(range(intervals[i], intervals[i + 1])):
                interval_dict[timestep] = i

        if self.verbose:
            print(interval_dict.items())

        return intervals, interval_dict

    def solve_mip(self, return_model: bool = False):
        '''
        Solve the mixed-integer programming problem for the optimal policy
        
        :param return_model: whether to return the model, defaults to False
        :type return_model: bool, optional
        :return: (policy, model)

        '''

        try:
            # Create new Gurobi MIP model
            m = gp.Model("mip_{}_{}_{}_{}_{}".format(len(self.arms), self.k, self.min_sel_frac, self.min_pull_per_pd,
                                                     self.interval_len))
            obj = gp.LinExpr()
            decision_vars = OrderedDict()
            _, interval_lookup = self.compute_interval_lookup()

            for i, arm in enumerate(self.arms):

                decision_vars[i] = list()
                obj_func_coeff = self.compute_obj_function_coefficients(arm=arm)

                # Placeholder to index into the arm's objective function coefficients at each action-timestep iteration
                counter = 0
                actions = list(range(self.n_actions))
                # Create decision vars x_i \in {0,1}, representing all timestep-action pairs (eg: i_t0_noPull, i_t0_pull, etc.)
                for t in range(0, self.horizon):
                    for action in actions:

                        if self.type == "linear":
                            xi = m.addVar(vtype=GRB.CONTINUOUS, 
                                          lb=0.0, 
                                          ub=1.0, 
                                          name="x_i{}_t{}_a{}_{}".format(i, t, action, self.type))
                            decision_vars[i].append(xi)
                        elif self.type == "integer":
                            xi = m.addVar(vtype=GRB.BINARY, name="x_i{}_t{}_a{}_{}".format(i, 
                                                                                           t, 
                                                                                           action, 
                                                                                           self.type))
                            decision_vars[i].append(xi)
                        else:
                            print("self.type must be either linear or integer")
                            break

                        # add c[x_i_t_a]*decision_variables[x_i_t_a] to our objective function
                        obj.addTerms(obj_func_coeff[counter], xi)
                        counter += 1

                    # Add constraint: we must take *exactly one* action for arm i at timestep t
                    m.addConstr(np.sum(decision_vars[i][self.n_actions * t:self.n_actions * (t + 1)]) == 1,
                                "c_{}_{}".format(i, t))

                # Add constraint: we must pull arm i >= min_sel_frac*horizon times over all timesteps
                # note: this step selection logic relies on zero-indexing
                if self.min_sel_frac > 0:
                    m.addConstr(
                        np.sum([x for i, x in enumerate(decision_vars[i]) if i % self.n_actions == self.pull_action]) >=
                        self.min_sel_frac * self.horizon, "c_{}_{}_minSelFrac".format(i, self.min_sel_frac))

                # Add constraint: we must pull arm i >= min_pull_per_pd (defaults to 0; in use, will be 1) for each interval
                # note: this step selection logic relies on zero-indexing
                if self.min_pull_per_pd > 0:
                    for interval in np.unique(
                            [v for k, v in interval_lookup.items() if k < self.horizon * self.n_actions]):
                        m.addConstr(np.sum([x for j, x in enumerate(decision_vars[i])
                                            if interval_lookup[j] == interval
                                            and j % self.n_actions == self.pull_action]) >= self.min_pull_per_pd,
                                    "c_{}_{}_periodicity".format(i, self.min_pull_per_pd))

            # Add  budget constraint: we can pull exactly k arms at each timestep
            for t in range(0, self.horizon):
                # Note: this selection logic for timestep-pull action indices relies on zero indexing
                pull_at_t_xis = [v[t * self.n_actions + self.pull_action] for _, v in decision_vars.items()]
                m.addConstr(np.sum(pull_at_t_xis) == self.k, "c_budget_k{}_t{}".format(self.k, t))

            # We want to maximize our objective function (sum of expected rewards)
            m.setObjective(obj, GRB.MAXIMIZE)

            # Optimize model
            m.optimize()

        except gp.GurobiError as e:
            self.error_log.exception('Error code ' + str(e.errno) + ': ' + str(e) + 'in MIPPolicy.solve_mip()')
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError:
            self.error_log.exception('Encountered an attribute error in MIPPolicy.solve_mip()')
            print('Encountered an attribute error')

        except:
            self.error_log.exception('Encounted an error in MIPPolicy.solve_mip()')

        policy = np.array([abs(v.x) for v in m.getVars()])
        policy = self.format_solution(policy)
        
        if return_model:
            return policy, m
        else:
            return policy
    
    def select_k_arms(self, 
                      t: int, 
                      arms: [RestlessArm] = None, 
                      k: int = None, 
                      **kwargs):
        if arms is None:
            arms = self.arms
        if arms != self.arms:
            warnings.warn('MathProgPolicy is not intended to be used on subsets of cohorts.')
        if k is None:
            k = self.k
   
        arms_to_pull = np.where(self.policy[:,t]==self.pull_action)[0][:k]
        return [arm.id for i, arm in enumerate(arms) if self.lookup_position([arm]) in arms_to_pull]

    def format_solution(self, policy: np.array):
        '''
        Reshape policy solution
        
        :param policy: n_arms*horizon*n_actions array
        :type policy: np.array
        :return: n_arms x horizon of 0 (no action) or 1 (action)
        :rtype: np.array

        '''

        soln = policy.reshape((-1, self.horizon, self.n_actions))
        formatted_policy = np.zeros((len(self.arms), self.horizon), dtype=int)
        for t in range(self.horizon):
            pulled_arms = np.where(soln[:, t, self.pull_action] == 1)[0]
            # if verbose:
            #     print(f'At time t={t}, pull arms {pulled_arms}')
            formatted_policy[pulled_arms, t] = 1

        return formatted_policy


    @staticmethod
    def validate_solution(policy: np.array, 
                          min_sel_frac: float, 
                          k: int, 
                          horizon: int):
        '''
        Validates a deterministic policy for pull budget and minimum selection constraints
        
        :param policy: n_arms x horizon policy
        :type policy: np.array
        :param min_sel_frac: minimum fraction of times an arm must be selected
        :type min_sel_frac: float
        :param k: pull budget
        :type k: int
        :param horizon: simulation horizon
        :type horizon: int
        :return: whether the solution is valid
        :rtype: bool

        '''
       
        # matrix of zeros and ones only
        assert (np.all((policy == 0) | (policy == 1)))

        # pull k arms at each timestep
        assert (np.all(np.sum(policy, axis=0) == k))

        # each arm is pulled at least min_sel_frac percent of the time
        for arm in range(np.shape(policy)[0]):
            pulled_times = len(np.where((policy[arm, :] == 1))[0])
            assert (min_sel_frac <= pulled_times / horizon)


if __name__ == "__main__":
    pass
