#!/usr/bin/env python3
import re


def get_counts(text):
  counts = { "failed" : 0, "passed" : 0, "skipped" : 0}
  for phrase in text.split(","):
    count, status = phrase.strip().split()
    counts[status] = int(count)
  return counts


def extract_test_summary(test_output):
  summary = {}
  if len(test_output) < 6:
    return summary
  summary["name"] = test_output[1]
  summary["command"] = test_output[3]
  for line in test_output:
    match = re.match(r"=+ (.*) in (.*) =+", line)
    if match:
      summary["counts"] = get_counts(match.group(1))
      summary["duration"] = match.group(2).split()[0]
  return summary


def extract_summaries(testing_output):
  summaries = []
  with open(testing_output) as f:
    test_output = []
    for line in f.readlines():
      if re.search("#"*100, line):
        summary = extract_test_summary(test_output)
        if summary: summaries.append(summary)
        test_output = []
      else:
        test_output.append(line.strip())
    summary = extract_test_summary(test_output)
    summaries.append(summary)
  return summaries


def print_summaries(summaries):
  print ("{:50s} : {:>10s} {:>10s} {:>10s} {:>14s}".format("TESTCASE", "PASSED", "FAILED", "SKIPPED", "DURATION"))
  print ("-"*100)
  total_passed, total_failed, total_skipped = 0,0,0
  for summary in summaries:
    name = summary["name"]
    counts = summary["counts"]
    num_passed = counts["passed"]
    num_failed = counts["failed"]
    num_skipped = counts["skipped"]
    duration = summary["duration"]
    num_passed_str = str(num_passed) if num_passed > 0 else ""
    num_failed_str = str(num_failed) if num_failed > 0 else ""
    num_skipped_str = str(num_skipped) if num_skipped > 0 else ""
    print ("{:50s} : {:>10s} {:>10s} {:>10s} {:>14s}".format(name, num_passed_str, num_failed_str, num_skipped_str, duration))
    total_passed += num_passed
    total_failed += num_failed
    total_skipped += num_skipped

  print ("-"*100)
  print ("{:50s} : {:10d} {:10d} {:10d}".format("TOTAL", total_passed, total_failed, total_skipped))


def main():
  testing_output = "test_9.log"
  summaries = extract_summaries(testing_output)
  print_summaries(summaries)


if __name__ == '__main__':
  main()
