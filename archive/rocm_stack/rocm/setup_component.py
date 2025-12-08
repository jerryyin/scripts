#! /usr/bin/python3

import subprocess
import argparse
import sys

supported_libraries = {
    "roctracer" : {
        "url" : "https://github.com/ROCm-Developer-Tools/roctracer",
        "clone_options" : [],
        },
    "hcc" : {
        "url" : "https://github.com/RadeonOpenCompute/hcc",
        "clone_options" : ["--recursive"],
        },
    "MLOpen" : {
        "url" : "https://github.com/AMDComputeLibraries/MLOpen",
        "clone_options" : [],
        },
}

def handle_clone(library, location):

    info = supported_libraries[library]
    
    url = info["url"]
    clone_cmd = ["git", "clone"] + info["clone_options"] +  [url, location]
    # print (clone_cmd)
    
    result = subprocess.run(clone_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(clone_cmd)))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
        
    return result.returncode


def handle_build_install(library, location, do_install):

    pass


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--location", default=None)
    parser.add_argument("--clone", action='store_true')
    parser.add_argument("--build", action='store_true')
    parser.add_argument("--install", action='store_true')
    args = parser.parse_args()
    # print (args)

    if args.library not in supported_libraries.keys() :
        print("\tNo support for library {}".format(args.library))
        sys.exit(1)
    
    if args.location is None:
        location = "/root/{}".format(args.library)
        print ("")
        print ("\trepo clone destination not specified for {}".format(args.library), end="")
        print (', defaulting to using "{}"'.format(location))
        print ("")
    
    if args.clone :
        handle_clone(args.library, location)


    if args.build or args.install :
        handle_build_install(args.library, location, args.install)
