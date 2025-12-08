#!/usr/bin/env python3

import shutil
import subprocess
import re
import json

def run_command(cmd, shell=True):
  """Structures for a variety of different test results.
    
  Args:
    cmd: Command to execute
    shell: True to use shell, false otherwise.

  Returns:
    Tuple of the command return value and the standard out in as a string.
"""
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, shell=shell)

  exit_code = None
  line = ''
  stdout = ''
  while exit_code is None or line:
    exit_code = p.poll()
    line = p.stdout.readline().decode('utf-8')
    stdout += line

  return exit_code, stdout


def _get_amd_gpu_info():
  """Returns gpu information using rocm-smi.

  Note: Assumes if the system has multiple GPUs, that they are all the same

  Returns:
    A dict containing gpu_driver_version, gpu_model and gpu_count or None if
    `rocm-smi` is not found or fails.
  """
  cmd = 'rocm-smi --json --showproductname --showdriverversion'
  exit_code, result = run_command(cmd)

  if exit_code != 0:
    logging.error('rocm-smi did not return as expected: %s', result)
    return None

  def get_gpu_driver_version(rocm_smi_output):
    return rocm_smi_output['system']['Driver version']

  def get_gpu_model(rocm_smi_output):
    gpu_model = ""
    for key, value in rocm_smi_output.items():
      if re.match("card[0-9]+", key):
        gpu_model = value['Card SKU']
    return gpu_model

  def get_gpu_count(rocm_smi_output):
    gpu_count = 0
    for key, value in rocm_smi_output.items():
      if re.match("card[0-9]+", key):
        gpu_count += 1
    return gpu_count


  rocm_smi_output= json.loads(result)
  print (rocm_smi_output)
  
  gpu_info = {}
  gpu_info['gpu_driver_version'] = get_gpu_driver_version(rocm_smi_output)
  gpu_info['gpu_model'] = get_gpu_model(rocm_smi_output)
  gpu_info['gpu_count'] = get_gpu_count(rocm_smi_output)

  print (gpu_info)
  # return gpu_info




_get_amd_gpu_info()
