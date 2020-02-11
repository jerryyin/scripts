#! /usr/bin/python3

import subprocess
import argparse
import sys

supported_libraries = {
    "roctracer" : {
        "url" : "https://github.com/ROCm-Developer-Tools/roctracer",
        },
}

def handle_clone(library, location):

    url = supported_libraries[library]["url"]
    clone_cmd = ["git", "clone", url, location]
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
        print ("\tsource/repo location not specified for {}".format(args.library))
        print ("\tdefaulting to using {}".format(location))
        print ("")
    
    if args.clone :
        handle_clone(args.library, location)


    if args.build or args.install :
        handle_build_install(args.library, location, args.install)
