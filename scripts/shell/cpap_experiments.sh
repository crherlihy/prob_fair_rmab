#!/bin/bash

## Args
# config, sid, ipaddress, Policy, local_r, Simulation, heuristic

## Initialize
date;hostname;pwd


source activate fair_rmab
config="./../../config/cpap_experiments.ini"


ipaddress=$(curl -s ifconfig.me)
echo "IP address: $ipaddress"
# echo "\nRunning on ${hostname}"

out=$(python ./../../src/Database/get_foreign_key_vals.py --verbose=False --db_ip="$ipaddress" -c="$config")
echo "out $out"
sid=$(echo $out | tail -n 1 | cut -d' ' -f2) 
echo "SID $sid" 

cd ./../slurm || exit
pwd

## Experiment-specific params
declare -a policies=("NoActPolicy" "RoundRobinPolicy" "RandomPolicy") 
declare -a localrs=("belief_identity" "mate21_concave")
declare -a heuristics=("first" "last" "random")
declare -a lbs=("0.1")
declare -a pctNonadhering=("0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
T=180
N=100

function get_n_nonadhering()
{
python - <<START
N=$1
pctNonadhering=$2
print(int(N*pctNonadhering))
START
}

i=1

for pct in "${pctNonadhering[@]}"; do
  nNonadhering=$(get_n_nonadhering "$N" "$pct")
  nGeneral=$(($N-$nNonadhering))

  ## Run
  # ProbFairPolicy
  for lb in "${lbs[@]}"; do
      echo "ProbFairPolicy; lb ${lb}; trial ${i}; N non-adhering ${nNonadhering}"
      sh ./cpap_experiment.sh "$config" "$sid" "$ipaddress" "ProbFairPolicy" "belief_identity" "Simulation" "None" "$nNonadhering" "$nGeneral"
      let "sid++"
      let "i++"
  done

  # basline TW, RAW
  for localr in "${localrs[@]}"; do
      echo "WhittleIndexPolicy; ${localr}; trial ${i} N non-adhering ${nNonadhering}"
      sh ./cpap_experiment.sh "$config" "$sid" "$ipaddress" "WhittleIndexPolicy" "$localr" "Simulation" "None" "$nNonadhering" "$nGeneral"
      let "sid++"
      let "i++"
  done

  # Baseline policies
  for policy in "${policies[@]}"; do
      echo "${policy}; trial ${i} N non-adhering ${nNonadhering}"
      sh ./cpap_experiment.sh "$config" "$sid" "$ipaddress" "$policy" "belief_identity" "Simulation" "None" "$nNonadhering" "$nGeneral"
      let "sid++"
      let "i++"
  done
done