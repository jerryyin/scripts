import sys
import os
import subprocess

def parse_hit(hit) :

    hit = hit.split("\n")

    first_line = hit[0].split("-")
    filename = first_line[0]
    line1 = first_line[2]
    
    second_line = hit[1].split(":")
    line_num = int(second_line[1])
    line2 = second_line[2]

    third_line = hit[2].split("-")
    line3 = third_line[2]

    return filename, line_num, line1, line2, line3

    
def parse_grep_output(grep_output):

    files_to_edit = {}
    
    content = grep_output.split("--\n")
    for hit in content :
        filename,line_num,line1,line2,line3 = parse_hit(hit)
        if line2 == "#define EIGEN_USE_HIP" :
            if (line1 == "#if TENSORFLOW_USE_ROCM") and (line3 == "#endif") :
                files_to_edit[filename] = "{},{}d".format(line_num-1, line_num+1)
            else :
                files_to_edit[filename] = "{}d".format(line_num)

    return files_to_edit


def run_sed(files_to_edit):

    for filename, lines in files_to_edit.items():
        tmp_filename ="file.tmp"
        os.system("sed -e '{}' {} > {}".format(lines, filename, tmp_filename))
        os.system("mv {} {}".format(tmp_filename, filename))


def main():

    grep_output = ""
    
    try :
        grep_output = subprocess.check_output(["grep", "-n", "-A1", "-B1", "#define EIGEN_USE_HIP", "-r", "."], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print (e.return_code)
    
    files_to_edit = parse_grep_output(grep_output)
    run_sed(files_to_edit);
    

if __name__=="__main__":
    main()
