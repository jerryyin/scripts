
from argparse import ArgumentParser
import sys
import re
import subprocess

def parse_command_line():
    """
    parse the command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("mangled_file")
    parser.add_argument("demangled_file")
    return parser.parse_args()

def get_mangled_names(mangled_file):
    """
    find all occurrences of C++ mangled names in the 'mangled_file' 
    and return the resulting set.
    """
    mangled_names = set();
    with open(mangled_file) as fp :

        contents = fp.read();

        mangled_name_pattern = r'ShaderName : (_Z[^ \t\n]*)'
        for match in re.finditer(mangled_name_pattern, contents):
            mangled_names.add(match.group(1))
            

    for name in mangled_names:
        print (name)
    return mangled_names


def demangle_names(mangled_names):
    """
    return the 'mangled' -> 'demangled' dictionary corresponding to
    the given set of C++ mangled names
    """
    demangled_names = {}

    for mangled_name in mangled_names :

        demangled_name = subprocess.run(["c++filt", mangled_name], stdout=subprocess.PIPE).stdout.decode("utf-8")
        demangled_names[mangled_name] = demangled_name
        
    return demangled_names


def write_updated_file(orig, updated, sr_dict):
    """
    Use the 'sr_dict' (which contains a mapping of 'search' --> 'replace' pairs) to
    update the contents of orig (occurences of 'search' are replaced by 'replace').

    the 'orig' file is not modified. The updated contents are written to 'updated'
    """

    with open(orig) as fp :
        contents = fp.read();
    
    for s,r in sr_dict.items():
        contents = contents.replace(s,r)

    with open(updated, "w") as fp :
        fp.write(contents)



def main():

    cmdline = parse_command_line()

    mangled_names = get_mangled_names(cmdline.mangled_file)

    demangled_dict = demangle_names(mangled_names)

    write_updated_file(cmdline.mangled_file, cmdline.demangled_file, demangled_dict)

    return 0

    
if __name__=='__main__':
    sys.exit(main())
