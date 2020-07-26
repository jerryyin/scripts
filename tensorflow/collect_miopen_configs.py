#!/usr/bin/env python3

import re
import subprocess
import sys
import random
from multiprocessing import Pool

def run_shell_command(shell_cmd, env_vars):
    all_env_vars = " ".join(["{}={}".format(key, value) for key, value in env_vars.items()])
    full_cmd = " ".join([all_env_vars, shell_cmd])
    print(full_cmd)
    # result = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    result = subprocess.run(full_cmd, shell=True)
    if result.returncode != 0:
        print("FAILED - {}".format(full_cmd))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
    return result


def collect_miopen_configs(filename):
    miopen_configs = set()
    with open(filename) as f:
        pattern = r'MIOpen\(HIP\): Command'
        for line in f.readlines():
            if re.match(pattern, line):
                miopen_configs.add(line.strip())
    # for config in miopen_configs:
    #     print (config)
    return miopen_configs


def get_miopen_driver_commands(rocm_path, miopen_configs):
    miopen_driver_commands = []
    for config in miopen_configs:
        pos = config.find("./bin/MIOpenDriver")
        if pos == -1:
            sys.exit(1)
        cmd = "{}/miopen/{}".format(rocm_path,config[pos+2:]) 
        miopen_driver_commands.append(cmd)
    return sorted(miopen_driver_commands)


def update_perf_db_using_one_gpu(gpu_id, miopen_driver_commands):
    env_vars = {
        "ROCR_VISIBLE_DEVICES" : gpu_id,
        "MIOPEN_FIND_ENFORCE" : 4,
    }
    for cmd in miopen_driver_commands:
        run_shell_command(cmd, env_vars)
    

def update_perf_db_using_all_gpus(num_gpus, miopen_driver_commands):
    
    random.shuffle(miopen_driver_commands)
    num_cmds = len(miopen_driver_commands)
    cmds_per_gpu = num_cmds // num_gpus
    remainder = num_cmds % num_gpus
    args = []
    pos = 0
    for gpu in range(num_gpus):
        num = cmds_per_gpu + (1 if (gpu < remainder) else 0)
        args.append([gpu, miopen_driver_commands[pos : pos + num]])
        pos += num

    with Pool(num_gpus) as p:
        p.starmap(update_perf_db_using_one_gpu, args)
        
    
if __name__ == '__main__':

    rocm_path = "/opt/rocm-3.5.0/"
    filename = "run_1.log"
    num_gpus = 1
    
    miopen_configs = collect_miopen_configs(filename)
    
    miopen_driver_commands = get_miopen_driver_commands(rocm_path, miopen_configs)

    update_perf_db_using_all_gpus(num_gpus, miopen_driver_commands)
