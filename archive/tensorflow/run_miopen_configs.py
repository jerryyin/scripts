#!/usr/bin/env python3

import re
import subprocess
import sys
import random
import os

# grep MIOpenDriver run_bs32_3.log | cut -d" " -f4- | sort | uniq

miopen_driver_configs="""
./bin/MIOpenDriver convfp16 -n 32 -c 192 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 192 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 192 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 3 -H 227 -W 227 -k 64 -y 11 -x 11 -p 0 -q 0 -u 4 -v 4 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 3 -H 227 -W 227 -k 64 -y 11 -x 11 -p 0 -q 0 -u 4 -v 4 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 384 -H 13 -W 13 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 384 -H 13 -W 13 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 384 -H 13 -W 13 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 384 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 384 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 384 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 64 -H 27 -W 27 -k 192 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 64 -H 27 -W 27 -k 192 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convfp16 -n 32 -c 64 -H 27 -W 27 -k 192 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver poolfp16 -M 0 -n 32 -c 192 -H 27 -W 27 -y 3 -x 3 -p 0 -q 0 -v 2 -u 2 -m max -F 1 -t 1
./bin/MIOpenDriver poolfp16 -M 0 -n 32 -c 192 -H 27 -W 27 -y 3 -x 3 -p 0 -q 0 -v 2 -u 2 -m max -F 2 -t 1
./bin/MIOpenDriver poolfp16 -M 0 -n 32 -c 256 -H 13 -W 13 -y 3 -x 3 -p 0 -q 0 -v 2 -u 2 -m max -F 1 -t 1
./bin/MIOpenDriver poolfp16 -M 0 -n 32 -c 256 -H 13 -W 13 -y 3 -x 3 -p 0 -q 0 -v 2 -u 2 -m max -F 2 -t 1
./bin/MIOpenDriver poolfp16 -M 0 -n 32 -c 64 -H 55 -W 55 -y 3 -x 3 -p 0 -q 0 -v 2 -u 2 -m max -F 1 -t 1
./bin/MIOpenDriver poolfp16 -M 0 -n 32 -c 64 -H 55 -W 55 -y 3 -x 3 -p 0 -q 0 -v 2 -u 2 -m max -F 2 -t 1
"""

def run_shell_command(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("FAILED - {}".format(full_cmd))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
    return result


def run_miopen_driver_configs():
  cfg_set = set()
  for cfg in miopen_driver_configs.split("\n"):
    if cfg :
      cfg_set.add(cfg)

  for cfg in cfg_set:
    print(cfg)
    os.chdir("/opt/rocm-5.0.0-9197/miopen")
    run_shell_command(cfg)

  
if __name__ == '__main__':
  run_miopen_driver_configs()
