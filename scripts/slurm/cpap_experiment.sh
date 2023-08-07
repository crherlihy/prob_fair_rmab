## Args
# config, sid, ipaddress, Policy, local_r, Simulation, heuristic, n_nonadhering, n_general

date;hostname;pwd

python ./../../src/run_simulation.py  --config_file="$1" --simulation_id="$2" --db_ip="${3}" --policy_name="$4" --local_r="$5" --simulation_type="$6" --heuristic="$7" --n_nonadhering="$8" --n_general="$9"