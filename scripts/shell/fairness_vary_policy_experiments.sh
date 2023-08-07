#!/bin/bash

## Args
# config, sid, ipaddress, Policy, local_r, Simulation, heuristic, lb, ub, nu

## Initialize
date;hostname;pwd

source activate fair_rmab
config="./../../config/fairness_vary_policy_experiments.ini"

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
declare -a nus=(18 10 6)
declare -a lbs=("0.05555555555" "0.1" "0.16666666666")
declare -a heuristics=("first" "last" "random")
T=180
i=1

## Run
# ProbFairPolicy
for lb in "${lbs[@]}"; do
    echo "ProbFairPolicy; lb ${lb}; trial ${i}"
    sh ./fairness_vary_policy_experiment.sh "$config" "$sid" "$ipaddress" "ProbFairPolicy" "belief_identity" "Simulation" "None" "$lb" "1.0" "${T}"
    let "sid++"
    let "i++"
done

# basline TW, RAW
for localr in "${localrs[@]}"; do
    echo "WhittleIndexPolicy; ${localr}; trial ${i}"
    sh ./fairness_vary_policy_experiment.sh "$config" "$sid" "$ipaddress" "WhittleIndexPolicy" "$localr" "Simulation" "None" "0.0" "1.0" "${T}"
    let "sid++"
    let "i++"
done

# heuristics
for nu in "${nus[@]}"; do
    for heuristic in "${heuristics[@]}"; do
        echo "heuristic ${heuristic}; nu ${nu}; trial ${i}"
        sh ./fairness_vary_policy_experiment.sh "$config" "$sid" "$ipaddress" "WhittleIndexPolicy" "belief_identity" "IntervalHeuristicSimulation" "$heuristic" "0.0" "1.0" "${nu}"
        let "sid++"
        let "i++"
    done
done

# Baseline policies
for policy in "${policies[@]}"; do
    echo "${policy}; trial ${i}"
    sh ./fairness_vary_policy_experiment.sh "$config" "$sid" "$ipaddress" "$policy" "belief_identity" "Simulation" "None" "0.0" "1.0" "${T}"
    let "sid++"
    let "i++"
done