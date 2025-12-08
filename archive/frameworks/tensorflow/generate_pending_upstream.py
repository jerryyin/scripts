#!/usr/bin/env python3

import subprocess
import argparse
import sys
import re

from datetime import date

from openpyxl import Workbook
from openpyxl.styles import Font


def run_shell_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
        sys.exit(result.returncode)
    return result


def generate_diff_stats(branch_1, branch_2):

    if branch_2 is not None:
        result = run_shell_command(["git", "diff", "--numstat", branch_1, branch_2, "--"])
    else :
        result = run_shell_command(["git", "diff", "--numstat", branch_1, "--"])

    diff_stats = result.stdout.decode().split('\n')
    diff_stats.pop() # last entry is an empty line

    return diff_stats


def display_num_diffs(branch_1, branch_2):

    diff_stats = generate_diff_stats(branch_1, branch_2)
    print("Num different files : ", len(diff_stats))
    for stat in diff_stats:
        num_lines_added, num_lines_deleted, filename = stat.split('\t')
        print (stat)


def display_diffs(branch_1, branch_2):

    diff_stats = generate_diff_stats(branch_1, branch_2)
    
    def show_this_diff(stat):
        num_lines_added, num_lines_deleted, filename = stat.split('\t')

        pattern = r'third_party'
        match = re.match(pattern, filename)
        if match is not None:
            return filename

        # num_lines_added = int(num_lines_added)
        # num_lines_deleted = int(num_lines_deleted)
        # if (num_lines_added <= 5) and (num_lines_deleted <= 5):
        #     return filename

        return None

    for stat in diff_stats:
        filename = show_this_diff(stat)
        if filename is not None:
            if branch_2 is not None:
                subprocess.run(["git", "diff", branch_1, branch_2, "--", filename])
            else :
                subprocess.run(["git", "diff", branch_1, "--", filename])


def write_to_excel(branch_1, branch_2):
    
    today = date.today()

    excel_file = "/common/pending_upstream_{}.xlsx".format(today.strftime("%y%m%d"))

    diff_stats = generate_diff_stats(branch_1, branch_2)
    
    workbook = Workbook()
    sheet = workbook.active

    sheet.append(["branch1:", branch_1])
    sheet.append(["branch2:", branch_2])
    sheet.append([])
    sheet.append(["command to display diff for <filename>:", "", "git diff {} {} -- <filename>".format(branch_1, branch_2)])
    sheet.append([])
    sheet.append(["# lines ADDED", "# lines DELETED", "Owner", "Filename"])
    for stat in diff_stats:
        num_lines_added, num_lines_deleted, filename = stat.split('\t')
        owner = ""
        sheet.append([num_lines_added, num_lines_deleted, owner, filename])

    for col in ['A', 'B', 'C']:
        sheet.column_dimensions[col].width = 20

    for cell in ['A1', 'A2', 'A4', 'A6', 'B6', 'C6', 'D6']:
        sheet[cell].font = Font(bold=True)

    workbook.save(filename=excel_file)


if __name__ == '__main__':

    branch_1 = "origin/master-with-triaged-diffs"
    branch_2 = "origin/develop-upstream"
    # branch_2 = None

    run_shell_command(["git", "fetch", "origin"])

    # display_num_diffs(branch_1, branch_2)
    # display_diffs(branch_1, branch_2)

    write_to_excel(branch_1, branch_2)
