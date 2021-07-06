#!/usr/bin/env python3

import re
import subprocess
import sys
import random
import os

miopen_driver_configs="""
./bin/MIOpenDriver conv -n 2 -c 48 -H 6 -W 6 -k 384 -y 2 -x 2 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 48 -F 4 -t 1
./bin/MIOpenDriver conv -n 2 -c 48 -H 6 -W 6 -k 384 -y 2 -x 2 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 48 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 48 -H 17 -W 17 -k 192 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 48 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 48 -H 17 -W 17 -k 192 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 48 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 48 -H 5 -W 5 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 48 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 48 -H 5 -W 5 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 48 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 8 -H 9 -W 27 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 8 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 8 -H 9 -W 27 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 8 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 84 -H 8 -W 8 -k 84 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 84 -F 4 -t 1
./bin/MIOpenDriver conv -n 4 -c 84 -H 8 -W 8 -k 84 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 84 -F 4 -t 1
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
    os.chdir("/opt/rocm-4.1.0-3940/miopen")
    run_shell_command(cfg)

  
if __name__ == '__main__':
  run_miopen_driver_configs()
