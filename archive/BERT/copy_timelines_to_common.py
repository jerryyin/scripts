#!/usr/bin/env python3

import sys
import glob
import os
import argparse
from shutil import copyfile

def copy_timelines(src_dir, dest_dir):
    print (src_dir, dest_dir)
    pwd = os.getcwd()
    os.chdir(src_dir)
    files = glob.glob("timeline*.json")
    os.chdir(pwd)
    for f in files:
        src_file = os.path.join(src_dir, f)
        dest_file = os.path.join(dest_dir, f)
        copyfile(src_file, dest_file)


def copy_all_timelines(train_dir, dest_dir):

    src_dir_gpu_0 = train_dir
    dest_dir_gpu_0 = os.path.join(dest_dir, "GPU_0")
    os.mkdir(dest_dir_gpu_0)
    copy_timelines(train_dir, dest_dir_gpu_0)

    for i in range(1,8):
        src_dir_gpu_i = os.path.join(train_dir, str(i))
        if os.path.isdir(src_dir_gpu_i):
            dest_dir_gpu_i = os.path.join(dest_dir, "GPU_{}".format(i))
            os.mkdir(dest_dir_gpu_i)
            copy_timelines(src_dir_gpu_i, dest_dir_gpu_i)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir")
    parser.add_argument("dest_dir")
    args = parser.parse_args()

    if not os.path.isdir(args.train_dir):
        print ("directory does not exist : {}".format(train_dir))
        
    if not os.path.isdir(args.dest_dir):
        print ("directory does not exist : {}".format(dest_dir))
        
    copy_all_timelines(args.train_dir, args.dest_dir)
