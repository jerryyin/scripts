#!/usr/bin/env python3

import os
import sys
import re
import json

scope_name_pattern = r"GpuDevice::ComputeHelper scheduled (.*) op (Conv2D|Conv2DBackpropInput|Conv2DBackpropFilter) on"
miopen_driver_command_pattern = r"MIOpen\(HIP\): Command \[LogCmdConvolution\] \./bin/MIOpenDriver"

def generate_scope_to_conv_config_map(log_file, json_file):

  lines_of_interest = []
  with open(log_file, "r", encoding='ISO-8859-1') as f:
    for line in f.readlines():
      if re.search(scope_name_pattern, line):
        lines_of_interest.append(line.strip())
      if re.search(miopen_driver_command_pattern, line):
        lines_of_interest.append(line.strip())

  # for i, line in enumerate(lines_of_interest):
  #   print(line)
  #   if i > 200:
  #     break

  scope_to_config = {}
  def add_scope_to_config_mapping(scope, config):
    prev_config = scope_to_config.get(scope, None)
    if prev_config and (config != prev_config):
      print ("Conflicting configs found for scope = {}".format(scope))
      print ("\tconfig1 = {}".format(prev_config))
      print ("\tconfig2 = {}".format(config))
      sys.exit(1)
    scope_to_config[scope] = config
    
  scope, config = None, None
  for i, line in enumerate(lines_of_interest):
    match = re.search(miopen_driver_command_pattern, line)
    if match :
      config = line[match.end():]
      # print (config)
    match = re.search(scope_name_pattern, line)
    if match :
      scope = line[match.start(1):match.end(1)]
      # print (scope)
    if scope and config:
      add_scope_to_config_mapping(scope, config)
      scope, config = None, None
      
  # print (len(scope_to_config.keys()))
  with open(json_file, "w") as f:
    json.dump(scope_to_config, f)
  

  
if __name__ == '__main__':

  generate_scope_to_conv_config_map("./resnet50v15_vlog3_miopen_log_cmd.log", "scope_to_config.json")
