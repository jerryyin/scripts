#! /usr/bin/env python3

import subprocess
import argparse
import sys
import os
import stat

config = {
    "vdi" : {
        "url" : "ssh://deven@gerrit-git.amd.com:29418/compute/ec/vdi",
        "clone_options" : [],
        "location" : "",
        "commit" : "amd-master",
        },
    "opencl" : {
        "url" : "ssh://deven@gerrit-git.amd.com:29418/compute/ec/opencl",
        "clone_options" : [],
        "location" : "",
        "commit" : "amd-master-next",
        },
    "hip" : {
        "url" : "ssh://deven@gerrit-git.amd.com:29418/compute/ec/hip",
        "clone_options" : [],
        "location" : "",
        "commit" : "amd-master-next",
        },
}


def run_shell_cmd(cmd):
    print (" ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
        sys.exit(result.returncode)
    return result.returncode


def update_config(location):
    
    if not os.path.exists(location):
        print ("directory {} does not exist....creating it".format(location))
        os.mkdir(location)
        
    for repo in ["hip", "vdi", "opencl"]:
        repo_location = os.path.join(location, repo)
        config[repo]["location"] = repo_location


def clone_repos():
    cwd = os.getcwd()
    for repo in ["hip", "vdi", "opencl"]:
        info = config[repo]
        url = info["url"]
        location = info["location"]
        commit = info["commit"]

        if os.path.exists(location):
            print("directory already exists : ", location)
            sys.exit(1)
            
        clone_cmd = ["git", "clone"] + info["clone_options"] +  [url, location]
        run_shell_cmd(clone_cmd)

        os.chdir(location)
        checkout_cmd = ["git", "checkout", commit]
        run_shell_cmd(checkout_cmd)
        
    os.chdir(cwd)
   

def make_executable(build_script):
    mode = os.stat(build_script).st_mode
    os.chmod(build_script, mode | stat.S_IEXEC)
    

def write_env_vars(f):
    f.write("export ROCclr_DIR={}\n".format(config["vdi"]["location"]))
    f.write("export OPENCL_DIR={}\n".format(config["opencl"]["location"]))
    f.write("export HIP_DIR={}\n".format(config["hip"]["location"]))


def create_build_vdi_script(rocm_path):
    build_script = os.path.join(config["vdi"]["location"], "build_vdi.sh")
    opencl_location = config["opencl"]["location"]
    vdi_build_dir = os.path.join(config["vdi"]["location"], "build")
    if not os.path.exists(build_script):
        with open(build_script, "w") as f:
            write_env_vars(f)
            f.write("rm -rf build && mkdir build\n")
            f.write("cd build\n")
            f.write("cmake \\\n")
            f.write("  -DROCM_PATH={} \\\n".format(rocm_path))
            f.write("  -DOPENCL_DIR={} \\\n".format(opencl_location))
            f.write("  -DUSE_COMGR_LIBRARY=ON \\\n")
            f.write("  -DCMAKE_PREFIX_PATH={} \\\n".format(vdi_build_dir))
            f.write("  -DCMAKE_INSTALL_PREFIX={} \\\n".format(rocm_path))
            f.write("  ..\n")
            f.write("make -j$(nproc)\n")
    else :
        print ("build script already exists : ", build_script)
        
    make_executable(build_script)

   
def create_build_opencl_script(rocm_path):
    pass
   

def create_build_hip_script(rocm_path):
    build_script = os.path.join(config["hip"]["location"], "build_hip.sh")
    vdi_build_dir = os.path.join(config["vdi"]["location"], "build")
    if not os.path.exists(build_script):
        with open(build_script, "w") as f:
            write_env_vars(f)
            f.write("rm -rf build && mkdir build\n")
            f.write("cd build\n")
            f.write("cmake \\\n")
            f.write("  -DROCM_PATH={} \\\n".format(rocm_path))
            f.write("  -DHIP_COMPILER=clang \\\n")
            f.write("  -DHIP_PLATFORM=rocclr \\\n")
            f.write("  -DCMAKE_PREFIX_PATH={} \\\n".format(vdi_build_dir))
            f.write("  -DCMAKE_INSTALL_PREFIX={} \\\n".format(rocm_path))
            f.write("  ..\n")
            f.write("make -j$(nproc)\n")
            # f.write("make -j$(nproc) install\n")
    else :
        print ("build script already exists : ", build_script)
        
    make_executable(build_script)


def create_build_scripts(rocm_path):
    create_build_vdi_script(rocm_path)
    create_build_opencl_script(rocm_path);
    create_build_hip_script(rocm_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default=None)
    parser.add_argument("--rocm_path", default=None)
    args = parser.parse_args()

    update_config(os.path.abspath(args.location))
    
    # # Do this outside of the docker container
    # clone_repos()

    # Do this inside the docker container
    create_build_scripts(args.rocm_path)

    

if __name__ == "__main__" :
    main()

# # build and copy the release build HIP runtime
# export ROCM_PATH=/opt/rocm-3.6.0
# cd ~/vdi_src
# export OPENCL_DIR="$(readlink -f opencl)"
# export ROCclr_DIR="$(readlink -f vdi)"
# export HIP_DIR="$(readlink -f hip)"
# mkdir -p /root/vdi_src/vdi/build && cd /root/vdi_src/vdi/build 
# rm -rf *
# cmake -DOPENCL_DIR=/root/vdi_src/opencl/ -DUSE_COMGR_LIBRARY=ON -DCMAKE_PREFIX_PATH="$ROCclr_DIR/build" .. -DCMAKE_INSTALL_PREFIX=$ROCM_PATH 
# make -j 
# mkdir -p /root/vdi_src/hip/build && cd /root/vdi_src/hip/build
# rm -rf *
# cmake -DHIP_COMPILER=clang -DHIP_PLATFORM=rocclr -DCMAKE_PREFIX_PATH="$ROCclr_DIR/build" .. -DCMAKE_INSTALL_PREFIX=$ROCM_PATH 
# make -j 
# ## copy the HIP shared library to the ROCm installation path
# cd /root/vdi_src/hip/build && cp lib/libamdhip64.so $ROCM_PATH/hip/lib/libamdhip64.so.3.6.30600
