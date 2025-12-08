#! /home/deven/anaconda3/bin/python3

# syscall    what it does
#
# read	     read bytes from a file descriptor (file, socket)
# write	     write bytes from a file descriptor (file, socket)
# open	     open a file (returns a file descriptor)
# close	     close a file descriptor
# fork	     create a new process (current process is forked)
# exec	     execute a new program
# connect    connect to a network host
# accept     accept a network connection
# stat	     read file statistics
# ioctl	     set I/O properties, or other miscellaneous functions
# mmap	     map a file to the process memory address space
# brk	     extend the heap pointer

import fileinput
import sys
import re


def main() :
    
    filename = sys.argv[1]


    # with open(filename) as f :
    #     for line in f.readlines() :
    #         new_line = re.sub(pointer_pattern, "<pointer>", line)
    #         print (new_line)

    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            
            # # skip sycalls that we do not care about
            # skip_pattern = "stat|fstat|lstat|getdents"
            # if re.match(skip_pattern, line) :
            #         continue

            # only keep the following syscalls
            keep_pattern = "open"
            if not re.search(keep_pattern, line) :
                    continue

            skip_pattern = "-1 ENOENT"
            if re.search(skip_pattern, line) :
                    continue
            
            # replace pointers with <pointer>
            pointer_pattern = "0x[0-9a-f]{1,16}"
            line = re.sub(pointer_pattern, "<pointer>", line)

            # replace pointers with <pointer>
            fd_pattern = "\) += [0-9]+"
            line = re.sub(fd_pattern, ")", line)

            print(line, end='')


if __name__ == "__main__" :
    main()
