'''ProbFair: probabilistic stationary policy
'''
import os
import itertools
import logging
import warnings
from math import floor
from collections import OrderedDict

import ray
import numpy as np
import scipy.optimize

#  to pip install gurobipy: python -m pip install -i https://pypi.gurobi.com gurobipy
import gurobipy as gp
import networkx as nx

from src.Policies.Policy import Policy
from src.Arms.Arm import Arm
from src.Arms.RestlessArm import RestlessArm

class ProbFairPolicy(Policy):
    '''ProbFair: probabilistic stationary policy
    '''
    def __init__(self, 
                 policy_seed: int,
                 horizon: int, 
                 arms: [Arm], 
                 k: int,
                 pull_action: int = 1,
                 prob_pull_lower_bound: float = 0.1, 
                 prob_pull_upper_bound: float = 1.0,
                 flag_piecewise_linear_approx: bool = False,
                 ncut: int = None,
                 epsilon: float = 0.0005,
                 error_log: logging.Logger = logging.getLogger('error_log'),
                 verbose: bool = False, 
                 **kwargs):
        '''
        :param policy_seed: int seed for rng in sample_from_pi. Note that this does not guarantee any comparability between (inherited) policies, this is purely for reproducibility.
        :param horizon: length of a simulation. Required kwarg for all policies; required if flag_piecewise_linear_approx is True
        :param arms: [Arm], required kwarg for all policies.
        :param k: budget, required kwarg for all policies.
        :param pull_action: int action that corresponds to a 'pull', defaults to 1
        :param prob_pull_lower_bound: lower bound probability of a pull ell/lb, float. Defaults to 0.1
        :param prob_pull_upper_bound: upper bound probability of a pull u/ub, float. Defaults to 1.0
        :param flag_piecewise_linear_approx: bool flag to run the piecewise linear approximation solver, default False
        :param ncut: optional int number of cuts if flag_piecewise_linear_approx is True
        :param epsilon: float approximation bound, defaults to 0.005
        :param error_log: logging.Logger error log, defaults to logging.getLogger('error_log')
        :param verbose: bool flag for additional print statments, defaults to False
        :param **kwargs: optional kwargs; unused

        '''

        Policy.__init__(self, 
                        horizon=horizon, 
                        arms=arms,
                        k=k, 
                        error_log=error_log,
                        verbose=verbose, 
                        **kwargs)
        
        self.seed = policy_seed # purely for saving in the db
        self.rng = np.random.default_rng(policy_seed)
        
        assert(prob_pull_lower_bound <= k/len(arms) <= prob_pull_upper_bound)
        
        self.pull_action = pull_action
        self.lb = prob_pull_lower_bound
        self.ub = prob_pull_upper_bound
        self.z = None # found by solve()
        self.epsilon = epsilon
        
        for arm in self.arms:
            arm.f_generator = self.f(arm)
        
        self.flag_piecewise_linear_approx = flag_piecewise_linear_approx
        self.ncut = ncut
        if self.flag_piecewise_linear_approx:
            self.ncut = ncut
            self.policy = self.solve_lp() # probabilistic policy of shape (N,)
            
        else:
            self.concavity, self.policy = self.initialize_policy()
          
    
    def compute_lpw_approx_f(self, arm: RestlessArm, npts: int = 5):
        '''
        Helper function computes arm.f(u) at npts for the piecewise linear approximation
        
        :param arm: arm
        :type arm: RestlessArm
        :param npts: number of cut points, defaults to 5
        :type npts: int, optional
        :return: (ptu, ptf)

        '''
        ptu = []
        ptf = []

        for i in range(npts):
            ptu.append(self.lb + (self.ub - self.lb) * i / (npts - 1))
            ptf.append(-1*arm.f(ptu[i]))

        return ptu, ptf

    def solve_lp(self, return_model: bool = False):
        '''
        Solve the linear program. For the piecewise linear approximation only.
        
        :param return_model: whether to return the model, defaults to False
        :type return_model: bool, optional
        :return: (policy, model)

        '''

        try:
            # Create new Gurobi MIP model
            m = gp.Model("probFairLP_{}_{}_{}_{}".format(len(self.arms), self.k, self.horizon, self.lb))
            #obj = gp.LinExpr()
            decision_vars = OrderedDict()

            # Each arm will have ncut \in Z^+ decision variables (ncut piece-wise approximations of f(p)
            # Iterate through the set of arms, and set up the decision-variable specific PWL objective function
            for i, arm in enumerate(self.arms):

                # Get the cut points for this arm
                ptu, ptf = self.compute_lpw_approx_f(arm, self.ncut)

                decision_vars[i] = list()

                # Create a decision variable (p_i) for this arm in [lower_bound, 1] and update the dictionary entry
                xi = m.addVar(self.lb, self.ub, name='arm_{}'.format(i))
                decision_vars[i].append(xi)

                # Set the piecewise linear approximation objective function for this x_i
                m.setPWLObj(xi, ptu, ptf)

            m.addConstr(np.sum([decision_vars[i] for i in range(len(self.arms))]) <= self.k, "c_k")

            # NOTE: https://www.gurobi.com/documentation/9.1/refman/objectives.html#subsubsection:PiecewiseObj
            # A variable can't have both a linear and a piecewise-linear objective term.
            # Setting a piecewise-linear objective for a variable will set the Obj attribute on that variable to 0.
            # Similarly, setting the Obj attribute will delete the piecewise-linear objective on that variable.
            # :. seems to be necessary to negate f(p) and minimize instead of maximizing

            #m.setObjective(obj, GRB.MAXIMIZE)

            m.optimize()

            #for v in m.getVars():
               # print('%s %g' % (v.varName, v.x))
               
            # if self.verbose:
            #     print('Obj: %g' % m.objVal)

        except gp.GurobiError as e:
            self.error_log.exception('Error code ' + str(e.errno) + ': ' + str(e) + 'in ProbFairPolicy.solve_lp()')
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError:
            self.error_log.exception('Encountered an attribute error in ProbFairPolicy.solve_lp()')
            print('Encountered an attribute error')

        except:
            self.error_log.exception('Encounted an error in ProbFairPolicy.solve_lp()')
           
        policy = np.array([v.x for v in m.getVars()])
        if self.verbose:
            # print(m.display())
            print(f'Probabilistic policy: {policy}')
        
        assert(self.validate_policy(policy))    
        
        if return_model:
            return policy, m
        else:
            return policy
        
    @staticmethod
    def f(arm: Arm):
        ''' Helper function to assign arm property f()
        
        :param arm: 
        :return: f: Callable arm-specific function f(p)
        '''
        
        a0 = arm.transition[0,0,1] # P_{0,1}^0
        a1 = arm.transition[1,0,1] - arm.transition[0,0,1] # P_{0,1}^1 - P_{0,1}^0
        b0 = 1 - arm.transition[0,1,1] + arm.transition[0,0,1] # 1 - P_{1,1}^0 + P_{0,1}^0
        b1 = arm.transition[0,1,1] - arm.transition[1,1,1] \
             - arm.transition[0,0,1] + arm.transition[1,0,1] # P_{1,1}^0 - P_{1,1}^1 - P_{0,1}^0 + P_{0,1}^1

        f = lambda p: (a0 + a1*p)/(b0 + b1*p)
        return f
    
    @staticmethod
    def get_f_concavity(arm: Arm) -> str:
        '''
        get the concavity/linear or convexity of one arm using h(p, x0, interval_len)
        :param arm: Arm that needs convexity information (we pass in the entire arm for error logging.)
            :param sign_h: Callable, sign is correlated with the second derivative of f
        :return: str classifying the arm as 'concave' or 'convex'

        '''
        a0 = arm.transition[0,0,1] # P_{0,1}^0
        a1 = arm.transition[1,0,1] - arm.transition[0,0,1] # P_{0,1}^1 - P_{0,1}^0
        b0 = 1 - arm.transition[0,1,1] + arm.transition[0,0,1] # 1 - P_{1,1}^0 + P_{0,1}^0
        b1 = arm.transition[0,1,1] - arm.transition[1,1,1] \
             - arm.transition[0,0,1] + arm.transition[1,0,1] # P_{1,1}^0 - P_{1,1}^1 - P_{0,1}^0 + P_{0,1}^1
             
        if b1 == 0:
            return "concave"
        elif b1 != 0 and a1 == 0:
            return "concave" if a0 == 0 else "convex"
        else:
            return "convex" if a0 > (a1*b0)/b1 else "concave"
        
    def initialize_policy(self): 
        '''
        Solve for the concavity of each arm and the policy
        
        :return: (concavity, policy)

        '''
        for arm in self.arms:
            arm.f = lambda p, arm=arm: arm.f_generator(p=p)
        concavity = [self.get_f_concavity(arm) for arm in self.arms]
        policy = self.solve(concavity_list=concavity)
        # for arm in self.arms:
        #     print(arm.f(self.lb), arm.f(self.ub))
        return concavity, policy
    
    def solve_p1(self, A: [Arm], n_B: int, candidate_z: float, seed: int):
        '''
        Find optimal z = min(k-lb|B|, ub|A|) and solve for p_i:
            minimize -1* sum_{i \in A} f(p_i) s.t. p_i \in [self.lb, self.ub] \forall i and \sum_{i \in A} = z
        :param A: list of concave arms, each with a function arm.f()
        :param n_B: int len of convex arms (in set B)
        :return: tuple (pis, z)
            :return pis: probabilistic policy for each arm in A
            :return z: quantity of budget used in this step
        '''

        # z = np.min((k-self.lb*n_B, self.ub*len(A)))
        
        J = lambda weights: -np.sum([arm.f(p=weights[i]) for i, arm in enumerate(A)])
        def sum_constraint(inputs: [float], z: float = candidate_z):
            # total must = 0 to be accepted
            total = z - np.sum(inputs)
            return total
        
        # w0 = np.array([self.rng.uniform(self.lb, self.ub) for arm in A])

        # the rng cannot be pickled so we need to use the deprecated approach here
        np.random.seed(seed=seed)
        w0 = np.array([np.random.uniform(self.lb, self.ub) for arm in A])

        res = scipy.optimize.minimize(J, 
                                      w0, 
                                      method='SLSQP',
                                      constraints=({'type': 'eq', 'fun': sum_constraint}),
                                      bounds=[[self.lb, self.ub] for arm in A])
        
        pis = res.x

        assert(np.isclose(candidate_z, np.sum(pis)))
        return pis


    def solve_p2(self, B: [Arm], k_prime: float):
        '''
        Solve for p_i given a set of strictly convex arms.
        
        :param B: list of strictly convex arms
        :param k_prime: budget constraint over the arms (k_prime = k-z, where z is from solve_P1)
        :return: tuple (B1, middle_set, z_prime, B2)
            :return B1: set of arms where p_i = lb
            :return middle_set: set of (one) arm where p_i = z_prime \in (lb, ub]
            :return z_prime: the middle arm's p_i value.
            :return B2: set of arms where p_i = ub

        '''

        n_B = len(B)
        gamma = floor((n_B * self.ub - k_prime) / (self.ub - self.lb))
        z_prime = k_prime - gamma * self.lb - (n_B - 1 - gamma) * self.ub

        if n_B - gamma - 1 > 0:
            sorted_arm_deltas = [(arm.id, arm.f(self.ub) - arm.f(self.lb)) for arm in B]
            sorted_arm_deltas.sort(key=lambda x: x[1])
            B2 = [arm_id for (arm_id, delta) in sorted_arm_deltas[-(n_B - int(gamma) - 1):]]

        else:
            B2 = []

        remaining_arms = [arm for arm in B if arm.id not in B2]

        #futures_round_b = [compute_delta.remote(arm, self.lb, z_prime) for arm in remaining_arms]
        #sorted_arm_deltas_b = ray.get(futures_round_b)
        sorted_arm_deltas_b = [(arm.id, arm.f(self.ub) - arm.f(self.lb)) for arm in remaining_arms]
        sorted_arm_deltas_b.sort(key=lambda x: x[1])

        B1 = [arm_id for (arm_id, delta) in sorted_arm_deltas_b[:int(gamma)]]

        if len(remaining_arms) > max(0,gamma):
            middle_arm_id, _ = sorted_arm_deltas_b[int(gamma)]
            middle_set = [middle_arm_id]

        else:
            middle_set = []

        return B1, middle_set, z_prime, B2

    def solve(self, concavity_list: [str]):
        '''
        Solve for the policy
        
        :param concavity_list: list of concavities (ordered by self.arm)
        :type concavity_list: [str]
        :return: probabilistic policy
        :rtype: np.array

        '''
        @ray.remote
        def solve_p1_p2_given_z(A: [Arm], B: [Arm], candidate_z: float, seed: int):
            policy = np.zeros([len(A) + len(B)])

            if len(A) > 0:
                A_pis = self.solve_p1(A=A, n_B=len(B), candidate_z=candidate_z, seed=seed)
                for (i, j) in zip(self.lookup_position(arms=A), np.arange(len(A))):
                    policy[i] = A_pis[j]

            if len(B) > 0:
                B1, middle_arm_ids, middle_pi, B2 = self.solve_p2(B=B, k_prime=self.k - candidate_z)
                for i in self.lookup_position(arm_ids=B1):
                    policy[i] = self.lb
                for i in self.lookup_position(arm_ids=middle_arm_ids):
                    policy[i] = middle_pi
                for i in self.lookup_position(arm_ids=B2):
                    policy[i] = self.ub

            arm_ids = [arm.id for arm in self.arms]
            policy_idx = self.lookup_position(arm_ids=arm_ids)
            obj_func_val = np.sum([arm.f(p=policy[policy_idx[i]]) for i, arm in enumerate(self.arms)])

            return candidate_z, obj_func_val, policy

        ray._private.services.address_to_ip = lambda _node_ip_address: _node_ip_address
        ray.init(num_cpus=int(os.cpu_count() - 2), ignore_reinit_error=True)

        A = [self.arms[i] for i in np.arange(len(concavity_list)) if concavity_list[i] == 'concave']
        B = [self.arms[i] for i in np.arange(len(concavity_list)) if concavity_list[i] == 'convex']

        assert (len(A) + len(B) == len(self.arms))

        if self.verbose:
            print(f'Group A (concave) has {len(A)} arms. Group B (convex) has {len(B)} arms.')

        # Search over all budget splits; choose the budget split and associated p* that maximizes E[R]
        end = min(round(self.k - self.lb * len(B), 20), self.ub*len(A))
        begin = self.lb*len(A)
        steps = floor(begin + (end-begin) / self.epsilon)
        start = begin

        grid_search_z_vals, stepsize = np.linspace(start, end, steps + 1, endpoint=True, retstep=True)
        seeds = self.rng.integers(low=0, high=10 ** 5,
                                  size=(len(grid_search_z_vals)))  # For reproducibility within ray.remote

        futures = [solve_p1_p2_given_z.remote(A, B, z, seed) for z, seed in zip(grid_search_z_vals, seeds)]
        results = ray.get(futures)

        minimizing_z_idx = np.argmax([obj_func_val for (z, obj_func_val, pis) in results])
        pis = results[minimizing_z_idx][2]
        self.z = results[minimizing_z_idx][0]
        
        assert (self.validate_policy(pis))
        ray.shutdown()
        return pis
        
    def validate_policy(self, policy: np.array):
        '''
        Validates a policy
        
        :param policy: probabilistic policy to validate
        :type policy: np.array
        :return: True if the policy is valid
        :rtype: bool

        '''
        assert(np.isclose(self.k, np.sum(policy), atol=self.epsilon))
        assert(np.all([(x >= self.lb or np.isclose(x, self.lb)) for x in policy]))
        assert (np.all([(x <= self.ub or np.isclose(x, self.ub)) for x in policy]))

        return True

    def sample_from_pi(self, policy = None, return_graph: bool = False):
        '''
        Sample from the probabilistic policy pi
        :param policy: probabilistic policy, default self.policy
        :param return_graph: bool whether to return G, default False
        :return: tuple (realized_policy, G)
            :return realized_policy: (N,) numpy array of 1/0 corresp to pull/don't pull
            :return G: graph used to sample from pi

        '''

        def simplify(alpha: float, beta: float) -> [float]:
            """
            Helper routine; the purpose is to randomly fix one of the variables X and  Y
            Two main properties must hold: 
                (1) E[Pr(X=1)] = alpha; E[Pr(Y=1)] = beta 
                (2) at least one of X and Y gets fixed
            """
            # If alpha = beta = 0, X and Y are both fixed at 0
            if np.isclose(alpha, 0) and np.isclose(beta, 0.0):
                return [0, 0]
            # If alpha = beta = 1, X and Y are both fixed at 1
            elif np.isclose(alpha, 1.0) and np.isclose(beta, 1.0):
                return [1, 1]
            
            # Case I. (originally Case II; re-ordered due to numpy precision issues):  alpha + beta = 1
            elif np.isclose(alpha + beta, 1.0):
                # With probability alpha we fix variable Y at 0 and X at 1; with probability beta we fix X at 0 and Y at 1
                out = self.rng.binomial(1, p=alpha)
                return [1, 0] if out else [0, 1]
            
            # Case II. (originally Case I; re-ordered due to numpy precision issues):  0 < alpha + beta < 1
            elif 0.0 < np.sum([alpha, beta]) < 1.0:
                # with probability alpha/(alpha + beta) fix variable Y at 0 and set Pr[X=1] = alpha + beta
                # with remaining probability of beta(alpha + beta), we fix X at 0 and set Pr[Y=1] = alpha + beta
                out = self.rng.binomial(1, p=alpha/(alpha+beta))
                return [alpha+beta, 0] if out else [0, alpha+beta]
            
            # Case III. 1 < alpha + beta < 2
            elif 1 < np.sum([alpha, beta]) < 2:
                
                # With probability (1-beta)/(2 - alpha - beta) we fix X at 1 and set Pr[Y=1] = alpha + beta - 1;
                # with remaining probability of (1-alpha)/(2-alpha-beta) we fix Y at 1 and set Pr[X=1] = alpha + beta-1
                alpha = 1.0 if np.isclose(alpha,1.0) else 0 if np.isclose(alpha, 0.0) else alpha
                beta = 1.0 if np.isclose(beta, 1.0) else 0 if np.isclose(beta, 0.0) else beta
                
                out = self.rng.binomial(1, p=(1-beta)/(2-alpha-beta))
                return [1, alpha+beta-1] if out else [alpha+beta-1, 1]
            return

        def pairing_tree(g: nx.DiGraph):
            '''
            Helper function to generate pairing tree
            
            :param g: graph
            :type g: nx.DiGraph
            :return: pairing tree

            '''

            def filter_node_preds(n: int):
                '''
                Return true if node n has no predecessors
                
                :param n: node
                :type n: int
                :return: whether n has any predecessors
                :rtype: bool

                '''
                return len(list(g.predecessors(n))) == 0

            # Let U denote the current set of nodes that have no parent (to start, U = L)
            U = nx.subgraph_view(g, filter_node=filter_node_preds)

            # If |U| == 1, we are done
            if U.number_of_nodes() == 1:
                return g

            # Else if |U| >= 2, group U into \floor{|U|/2} pairs plus at most one more element in an arbitrary way
            elif U.number_of_nodes() >= 2:

                A = self.rng.choice(a=[node for node in U.nodes()], 
                                    size=floor(U.number_of_nodes()/2),
                                    p=np.repeat(1/U.number_of_nodes(), U.number_of_nodes()), 
                                    replace=False)

                B = [node for node in U.nodes() if node not in A]

                pairs = itertools.zip_longest(A,B)
                U_new = nx.DiGraph()

                # For each of the \floor{|U|/2} pairs, create a new node, and make the two elements in the pair be children of this new node
                for pair in pairs:

                    # Here, we handle the case where |U| % 2 != 0. We can't call simplify with only one child node
                    # So, we need to just insert 1 child node, create a parent node for this child, insert an edge, and
                    # propagate (unmodified, unsimplified) p_i to this new parent node.
                    # For the pair element that is None, -1 is used as placeholder in parent_node id creation
                    # The `orig_id` is set to  whichever  pair element is NOT None, for eventual recovery of the integral action.
                    if pair[0] is None:
                        U_new.add_nodes_from([(pair[1], U.nodes()[pair[1]])])
                        U_new.add_nodes_from([("parent_{}_{}".format(-1, pair[1]),
                                               {"orig_id": U.nodes()[pair[1]]["orig_id"], 
                                                "p_i": U_new.nodes()[pair[1]]['p_i'],
                                                "z": None})])
                        U_new.add_edges_from([("parent_{}_{}".format(-1,pair[1]),  pair[1])])

                    elif pair[1] is None:
                        U_new.add_nodes_from([(pair[0], U.nodes()[pair[0]])])
                        U_new.add_nodes_from([("parent_{}_{}".format(pair[0], -1),
                                               {"orig_id":  U.nodes()[pair[0]]["orig_id"], 
                                                "p_i": U_new.nodes()[pair[0]]['p_i'],
                                                "z": None})])
                        U_new.add_edges_from([("parent_{}_{}".format(pair[0],-1), pair[0])])


                    else:
                        U_new.add_nodes_from([(p, U.nodes()[p]) for p in pair if p in U.nodes()])
                        U_new.add_nodes_from([("parent_{}_{}".format(pair[0], pair[1]),
                                               {"orig_id": "parent_{}_{}".format(pair[0], pair[1]), "p_i": None,
                                                "z": None})])
                        U_new.add_edges_from([("parent_{}_{}".format(pair[0], pair[1]), p) for p in pair])

                        # Each internal node u at height 1 runs simplify on the random variables represented by its two children
                        # one of the rand. vars gets fixed and other is sent to u's parent
                        x_i, y_i = simplify(U_new.nodes()[pair[0]]['p_i'], U_new.nodes()[pair[1]]['p_i'])

                        U_new.nodes()[pair[0]]['z'] = x_i
                        U_new.nodes()[pair[1]]['z'] = y_i
                        U_new.nodes()["parent_{}_{}".format(pair[0], pair[1])]["p_i"] = x_i if y_i in [0, 1] else y_i

                        # since values propagate up the tree for "fixing" in subsequent recursive calls, we need to propagate up the names of the variables that remain unfixed
                        U_new.nodes()["parent_{}_{}".format(pair[0], pair[1])]["orig_id"] = int(
                            U_new.nodes()[pair[0]]['orig_id']) if y_i in [0, 1] else \
                            int(U_new.nodes()[pair[1]]['orig_id'])

                # Since U_new is not a view, we need to compose the input graph with this new graph
                # any nodes (and their attributes) that existed already in g will be updated based on their values in U_new
                h = nx.compose(g, U_new)
                return pairing_tree(h)

        if policy is None:
            policy = self.policy
        G = nx.DiGraph()

        # Start with the label set L = {(1,p_1), (2, p_2) ... (t, p_t)} as the leaf set
        G.add_nodes_from([(i, {"orig_id": i, "p_i": x, "z": None}) for i,x in enumerate(policy)])

        # Call the pairing tree function to get a sample from the probability distribution (pi) that satisfies desired properties
        g_prime = pairing_tree(G)

        realized_policy_dict = OrderedDict(sorted({g_prime.nodes()[n]['orig_id']: g_prime.nodes()[n]['z'] 
                                                   for n in g_prime.nodes() 
                                                   if isinstance(g_prime.nodes()[n]['orig_id'], int) 
                                                   and g_prime.nodes()[n]['z'] in [0,1]}.items()))
        
        # since OrderedDict preserves order, we can use this along with policy.lookup_position()
        realized_policy = np.array(list(realized_policy_dict.values()))
        
        if return_graph:
            return realized_policy, G
        else:
            return realized_policy

    def select_k_arms(self, arms: [RestlessArm] = None, k: int = None, **kwargs):
        '''
        Select k arms to pull
        
        :param arms: group of arms, defaults to self.arms if None
        :type arms: [RestlessArm], optional
        :param k: budget, defaults to None
        :type k: int, optional
        :param **kwargs: unused kwargs
        :return: list of arm ids to pull
        :rtype: [int]

        '''
        if arms is None:
            arms = self.arms
        if arms != self.arms:
            warnings.warn('ProbFairPolicy is not intended to be used on subsets of cohorts.')
        if k is None:
            k = self.k

        realized_policy = self.sample_from_pi()
        arms_to_pull = np.where(realized_policy == self.pull_action)[0][:k]
        
        # Action logging requires arm.id because arm.id might not map sequentially to the natural numbers
        return [arm.id for i, arm in enumerate(arms) if self.lookup_position([arm]) in arms_to_pull]


if __name__ == "__main__":
    pass
