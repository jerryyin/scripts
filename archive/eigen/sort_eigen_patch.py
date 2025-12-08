#! /usr/bin/python

import sys
import re

if (len(sys.argv) != 4) :
    print ("USAGE: ")
    sys.exit(1)


orig_patch = sys.argv[1]
new_patch = sys.argv[2]
sorted_patch = sys.argv[3]
    
def get_diff_file(line):

    diff_file = None
    if re.match("diff -Naur", line) :
        diff_file = line.split()[-1].split("/")
        diff_file = "/".join(diff_file[1:])
    elif re.match("diff --git", line) :
        diff_file = line.split()[-1].split("/")
        diff_file = "/".join(diff_file[1:])
    return diff_file


orig_diff_files = []
with open(orig_patch) as fp :
    for line in fp.readlines() :
        diff_file = get_diff_file(line)
        if diff_file is not None :
            orig_diff_files.append(diff_file)
    

new_diffs = {}
with open(new_patch) as fp :
    diff_file = None
    diff_contents = ""
    for line in fp.readlines() :
        tmp = get_diff_file(line)
        if tmp is not None :
            if diff_file is not None :
                new_diffs[diff_file] = diff_contents
                diff_contents = ""
            diff_file = tmp
        diff_contents += line
    new_diffs[diff_file] = diff_contents


def is_unit_test_file(f):
    if re.match("test/", f) or re.match("unsupported/test/", f):
        return True
    return False
    

with open(sorted_patch, "w") as fp :

    for orig_file in orig_diff_files :
        if orig_file in new_diffs.keys() :
            #print(orig_file)
            fp.write(new_diffs[orig_file])
        else :
            print ("Orig diff missing in new patch : ", orig_file)

    for new_file in new_diffs.keys() :
        if new_file not in orig_diff_files :
            if is_unit_test_file(new_file) :
                #print ("New diff missing in original patch : {} : (UNIT TEST FILE)".format(new_file))
                pass
            else :
                print ("New diff missing in original patch : ", new_file)
                fp.write(new_diffs[new_file])
            
