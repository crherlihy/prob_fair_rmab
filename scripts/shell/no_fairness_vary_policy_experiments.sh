#!/bin/bash

## Initialize
date;hostname;pwd

source activate fair_rmab
config="./../../config/no_fairness_vary_policy_experiments.ini"

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
declare -a policies=("ProbFairPolicy" "WhittleIndexPolicy" "NoActPolicy" "RoundRobinPolicy")

i=1

## Run
for policy in "${policies[@]}"; do
    echo "$policy; Running trial ${i} ${config}"
    sh ./no_fairness_vary_policy_experiment.sh "$config" "$sid" "$ipaddress" "$policy"
    let "sid++"
    let "i++"
done

