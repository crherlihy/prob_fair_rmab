## Planning to Fairly Allocate: Probabilistic Fairness in the Restless Bandit Setting

This repository contains the source code required to reproduce the analysis and experiments presented in the paper ["Planning to Fairly Allocate: Probabilistic Fairness in the Restless Bandit Setting"](https://arxiv.org/abs/2106.07677) (to appear at KDD 2023).

----
### Set-up

#### Conda environment
All dependencies are specified in the conda environment `fair_rmab`.

`conda env create -f environment.yml`

`conda activate fair_rmab`

#### Database
Simulation results are stored in database on a [MariaDB Community server](https://mariadb.org/). Download and installation instructions for different operating systems are [located on their website](https://mariadb.com/kb/en/getting-installing-and-upgrading-mariadb/). The command to create a database named `prob_fair` is `CREATE DATABASE prob_fair;`.

To start the server, run `mysql.server start` from the command line. To interact with the database directly from the command line, run `mysql -u <username> -p <password>`. To connect the key python scripts to the database, fill in the `[database]` section of the configuration file.

Before running a simulation, run `./src/Database/create.py` to create all database tables.

#### Configuration files

`./config/*.ini`: Defines paths and experiment-specific hyper-parameters.

- To **replicate** an experiment, update the project `root_dir` location in the`[paths]` section (or the `[database]` section, see above) if needed, but otherwise use the associated config file as-is.

- To run a **new experiment**, you can create a new config file ( `./configs/example_simulations.ini` is provided as a guide).
----

### Replication of experiments

To replicate experiments (Section 5 & Appendix E), from the project root directory: `cd ./scripts/shell/`, then:

- `sh fairness_vary_policy_experiments.sh`: `ProbFair` versus fairness-aware alternatives.

- `sh vary_cohort_composition_experiments.sh`: `ProbFair` evaluated on a breadth of synthetically generated cohorts.

- `sh cpap_experiments.sh`: `ProbFair` evaluated on the CPAP dataset.

- `sh no_fairness_vary_policy_experiments`: `ProbFair` and the price of state agnosticism.

#### Key python scripts:

- Simulations are launched by the script `run_simulation.py` from configuration and command line arguments (see the code for examples). The main driver there is `run_simulation()`, which initializes `Cohort`, `Policy`, and `Simulation` classes, runs a simulation, and saves results to the database. \
\
When running multiple simulations, it is possible to parallelize at the cohort level, policy level, and at the iteration level (e.g., when bootstrapping). The recommended approach is to modify the scripts contained in `./scripts/slurm` to be `sbatch` files, and update the calls in `./scripts/shell` accordingly.

- `Analysis/generate_figures.py` computes key results from the simulations, such as expected reward, intervention benefit (IB) and Earth Mover's Distance (EMD).

#### Selected references:

- Policy class `WhittleIndexPolicy.py` is our own implementation of `Threshold Whittle` and `Risk-Aware Whittle` algorithms (Mate et. al. [2020](https://papers.nips.cc/paper/2020/file/b460cf6b09878b00a3e1ad4c72344ccd-Paper.pdf), [2021](https://dl.acm.org/doi/10.5555/3463952.3464057)).
- Cohort class `CPAPCohort.py` utilizes CPAP data from Kang et. al. [2013](https://pennstate.pure.elsevier.com/en/publications/markov-models-for-treatment-adherence-in-obstructive-sleep-apnea) and [2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4669975/)
----

### Contributors:
Contributors to this repo include: [Christine Herlihy](https://github.com/crherlihy), [Aviva Prins](https://github.com/avivaprins), and [Daniel Smolyak](https://github.com/dsmolyak). 

### Citation information:
If you find this code useful, please cite the following paper:
```
@conference{herlihy23planning,
title={{Planning to Fairly Allocate: Probabilistic Fairness in the Restless Bandit Setting}},
author={Herlihy, Christine and Prins, Aviva and Srinivasan, Aravind and Dickerson, John P.},
year=2023,
booktitle=KDD,
note={Full version: arXiv:2106.07677},
}
```