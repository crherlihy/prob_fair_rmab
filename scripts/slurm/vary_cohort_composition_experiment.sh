#!/bin/bash
date;hostname;pwd

python ./../../src/run_simulation.py  --config_file="$1" --simulation_id="$2" --cohort_name="$3" --horizon="$4" --policy_name="$5" --local_r="$6" --simulation_type="$7" --heuristic="$8" --interval_len="$9" --prob_pull_lower_bound="${10}" --db_ip="${12}" --n_convex="${13}" --n_concave="${14}"

