#!/usr/bin/env python3

import sys
import argparse
import subprocess
import os
import shutil

def add_args():
  parser = argparse.ArgumentParser(description="Utility make conflict a seperate merge.")
  parser.add_argument("--fname")
  parser.add_argument("--cmp")
  args = parser.parse_args()
  return args

def open_and_sort(fname):
  file = open(fname, 'r')
  params = file.read()
  paramsList = params.split(" ")
  paramsList.sort()
  file.close()
  return paramsList

def main():
  args = add_args()
  original = open_and_sort(args.fname)
  new = open_and_sort(args.cmp)
  if new == original:
    print("same")
  else:
    print("different")


if __name__ == "__main__":
  exit(main())
