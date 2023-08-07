#!/bin/bash

## Initialize
date;hostname;pwd

conda activate fair_rmab
config="./../../config/example_simulations.ini"

ipaddress=$(curl -s ifconfig.me)
echo "IP address: $ipaddress"
# echo "\nRunning on ${hostname}"

out=$(python ./../../src/Database/get_foreign_key_vals.py --config_file=$config --verbose=False)
echo "out $out"
sid=$(echo $out | tail -n 1 | cut -d' ' -f2) 
echo "SID $sid" 


## Run
python3 ./../../src/run_simulation.py --config_file=$config --simulation_id=$sid
let "sid++"