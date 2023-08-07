#!/bin/bash

## Initialize
date;hostname;pwd

source activate fair_rmab
config="./../../config/vary_cohort_composition_experiments.ini"

ipaddress=$(curl -s ifconfig.me)
echo "IP address: $ipaddress"
# echo "\nRunning on ${hostname}"

out=$(python ./../../src/Database/get_foreign_key_vals.py --verbose=False --db_ip="$ipaddress" -c="$config")
echo "out $out"
sid=$(echo $out | tail -n 1 | cut -d' ' -f2)
echo "SID $sid"

cd ./../slurm || exit
pwd

#Synthetic cohort, 100% convex, 75% convex, 50% convex, 25% convex, 0% convex
#ProbFair, (+ NoAct, TW for intervention benefit)
#Fix (ell, u) = (0.1, 1.0)
declare -a steps=("experiments")
declare -a policies=("NoActPolicy" "WhittleIndexPolicy" "RoundRobinPolicy")
declare -a cohorts=("random")
declare -a pctConvex=("0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
declare -a horizons=(180)
ell="0.1"
N=100

function get_n_convex()
{
python - <<START
N=$1
pct_convex=$2
print(int(N*pct_convex))
START
}

for step in "${steps[@]}"; do
  # Loop over horizons
  for T in "${horizons[@]}"; do
    # Loop over cohorts
    for cohort in "${cohorts[@]}"; do
      # For each cohort, loop over percent_convex values
      for pct in "${pctConvex[@]}"; do
        nConvex=$(get_n_convex "$N" "$pct")
        nConcave=$(($N-$nConvex))

        # For each (horizon, cohort, pct_convex), loop over non-ProbFair policies
        for policy in "${policies[@]}"; do
          echo "cohort: $cohort; n_convex: $nConvex, n_concave $nConcave, horizon: $T; $policy; Running trial ${sid}"
          sh ./vary_cohort_composition_experiment.sh "$config" "$sid" "$cohort" "$T" "$policy" "belief_identity" "Simulation" "None" "$T" "-1.0" "$step" "$ipaddress" "$nConvex" "$nConcave"
          (( "sid++" ))
        done

        # Then, for the current (horizon, cohort, pct_convex) combination, run ProbFair with (ell, uuu) = (0.1, 1.0)
        echo "cohort: $cohort; n_convex: $nConvex, n_concave $nConcave, horizon: $T; ProbFairPolicy; Running trial ${sid}"
        sh ./vary_cohort_composition_experiment.sh  "$config" "$sid" "$cohort" "$T" 'ProbFairPolicy' "belief_identity" "Simulation" "None" "$T" "$ell" "$step" "$ipaddress" "$nConvex" "$nConcave"
        (( "sid++" ))
      done
    done
  done
done