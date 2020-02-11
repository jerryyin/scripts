#! /usr/bin/python3

import subprocess
import re



def get_alexnet_cmds():
    return """
./bin/MIOpenDriver convbfp16 -n 512 -c 192 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 192 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 192 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 3 -H 227 -W 227 -k 64 -y 11 -x 11 -p 0 -q 0 -u 4 -v 4 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 3 -H 227 -W 227 -k 64 -y 11 -x 11 -p 0 -q 0 -u 4 -v 4 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 384 -H 13 -W 13 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 384 -H 13 -W 13 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 384 -H 13 -W 13 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 384 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 384 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 384 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 64 -H 27 -W 27 -k 192 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 64 -H 27 -W 27 -k 192 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 512 -c 64 -H 27 -W 27 -k 192 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
"""

def get_googlenet_cmds():
    return """
./bin/MIOpenDriver convbfp16 -n 32 -c 112 -H 14 -W 14 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 112 -H 14 -W 14 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 112 -H 14 -W 14 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 28 -W 28 -k 192 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 28 -W 28 -k 192 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 28 -W 28 -k 192 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 144 -H 14 -W 14 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 144 -H 14 -W 14 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 144 -H 14 -W 14 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 16 -H 14 -W 14 -k 48 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 16 -H 14 -W 14 -k 48 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 16 -H 14 -W 14 -k 48 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 16 -H 28 -W 28 -k 32 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 16 -H 28 -W 28 -k 32 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 16 -H 28 -W 28 -k 32 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 14 -W 14 -k 320 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 14 -W 14 -k 320 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 14 -W 14 -k 320 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 7 -W 7 -k 320 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 7 -W 7 -k 320 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 7 -W 7 -k 320 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 28 -W 28 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 7 -W 7 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 7 -W 7 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 7 -W 7 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 24 -H 14 -W 14 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 24 -H 14 -W 14 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 24 -H 14 -W 14 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 28 -W 28 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 3 -H 225 -W 225 -k 64 -y 7 -x 7 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 3 -H 225 -W 225 -k 64 -y 7 -x 7 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 14 -W 14 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 14 -W 14 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 14 -W 14 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 14 -W 14 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 14 -W 14 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 14 -W 14 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 28 -W 28 -k 96 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 28 -W 28 -k 96 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 28 -W 28 -k 96 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 48 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 48 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 48 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 480 -H 14 -W 14 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 112 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 112 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 112 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 144 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 144 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 144 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 24 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 24 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 24 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 14 -W 14 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 528 -H 14 -W 14 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 56 -W 56 -k 192 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 56 -W 56 -k 192 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 56 -W 56 -k 192 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 832 -H 7 -W 7 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 14 -W 14 -k 208 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 14 -W 14 -k 208 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 14 -W 14 -k 208 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
"""

def get_resnet50_cmds():
    return """
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 64 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
"""

def get_inception3_cmds():
    return """
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 128 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1280 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 160 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 160 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 160 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 160 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 160 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 160 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 2048 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 288 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 384 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 384 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 384 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 384 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 384 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 384 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 448 -H 8 -W 8 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 448 -H 8 -W 8 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 448 -H 8 -W 8 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 48 -H 35 -W 35 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 48 -H 35 -W 35 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 48 -H 35 -W 35 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 768 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 80 -H 73 -W 73 -k 192 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 80 -H 73 -W 73 -k 192 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 80 -H 73 -W 73 -k 192 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
"""

def get_inception4_cmds():
    return """
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1024 -H 17 -W 17 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 73 -W 73 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 73 -W 73 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 160 -H 73 -W 73 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 224 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 224 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 17 -W 17 -k 224 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 35 -W 35 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 71 -W 71 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 71 -W 71 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 192 -H 71 -W 71 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 17 -W 17 -k 256 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 17 -W 17 -k 256 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 17 -W 17 -k 256 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 35 -W 35 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 35 -W 35 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 224 -H 35 -W 35 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 17 -W 17 -k 256 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 17 -W 17 -k 256 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 17 -W 17 -k 256 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 17 -W 17 -k 320 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 17 -W 17 -k 320 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 256 -H 17 -W 17 -k 320 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 320 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 320 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 320 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 35 -W 35 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 448 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 448 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 384 -H 8 -W 8 -k 448 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 448 -H 8 -W 8 -k 512 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 448 -H 8 -W 8 -k 512 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 448 -H 8 -W 8 -k 512 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 512 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 147 -W 147 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 147 -W 147 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 147 -W 147 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 64 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 64 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 64 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 64 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 64 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 64 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 96 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 96 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 64 -H 73 -W 73 -k 96 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1
./bin/MIOpenDriver convbfp16 -n 32 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
"""


def run_all():

    # cmds = get_alexnet_cmds()
    cmds = get_googlenet_cmds()
    # cmds = get_resnet50_cmds()
    # cmds = get_inception3_cmds()
    # cmds = get_inception4_cmds()
    
    for i, cmd in enumerate(cmds.split("\n")):
        if cmd :
            driver_cmd = " && ".join(["cd /root/miopen/build", cmd])
            print (driver_cmd)
            driver_output = subprocess.run(driver_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not re.search("Verifies on CPU and GPU", driver_output.stdout.decode()):
                print ("```")
                print ("FAIL")
                print ("")
                print (driver_output.stdout.decode())
                print ("```")
                print ("")
                print ("")


run_all()

        

