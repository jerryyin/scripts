#!/usr/bin/env python3
"""
MIOpenDriver Command Processor using multiprocessing

This script processes a file containing MIOpenDriver commands,
executes them across multiple GPUs using separate processes,
and collects the results into a CSV file with the specified format.
The order in the output CSV matches the order in the input file.
"""

import argparse
import subprocess
import re
import csv
import os
import multiprocessing
from typing import List, Dict, Optional


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Process MIOpenDriver commands and collect results.")
    parser.add_argument("--input", required=True, help="Input file containing MIOpenDriver commands")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--miopendriver-path",
        default="/opt/rocm/bin/MIOpenDriver",
        help="Path to MIOpenDriver (default: /opt/rocm/bin/MIOpenDriver)",
    )
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")

    return parser.parse_args()


def execute_command(command: str) -> str:
    """
    Execute a shell command and return its output.

    Args:
        command: The command to execute

    Returns:
        str: Command output (stdout and stderr combined)

    Raises:
        subprocess.SubprocessError: If command execution fails
    """
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Warning: Command exited with code {process.returncode}")
        print(f"Command: {command}")
        if stderr:
            print(f"Error: {stderr}")

    # Combine stdout and stderr for parsing
    return stdout + stderr


def parse_output(command: str, output: str, cmd_index: int) -> Dict:
    """
    Parse the output of a MIOpenDriver command execution.

    Args:
        command: The original command
        output: The command output to parse
        cmd_index: The original index of the command in the input file

    Returns:
        Dict: Parsed information from the output (as dictionary for multiprocessing)
    """
    # Using dictionary instead of CommandResult class for easier serialization
    result = {
        "command": command,
        "direction": "",
        "algorithm": -1,
        "solution": "",
        "name": "",
        "gflops": 0.0,
        "time_ms": 0.0,
        "error": 0.0,
        "index": cmd_index,  # Store the original index
    }

    # Parse direction
    if "Backward Weights Conv" in output:
        result["direction"] = "Backward Weights"
    elif "Backward Data Conv" in output:
        result["direction"] = "Backward Data"
    elif "Forward Conv" in output:
        result["direction"] = "Forward"

    # Parse algorithm
    algo_match = re.search(r"Algorithm: (\d+)", output)
    if algo_match:
        result["algorithm"] = int(algo_match.group(1))

    # Parse solution
    solution_match = re.search(r"Solution: ([^\n]+)", output)
    if solution_match:
        result["solution"] = solution_match.group(1)

    # Parse stats lines
    stats_lines = [line.strip() for line in output.split("\n") if line.strip().startswith("stats:")]

    if len(stats_lines) >= 2:
        # Parse field names (first stats line)
        header_line = stats_lines[0][7:].strip()  # Remove "stats: " prefix
        field_names = [field.strip(" ") for field in header_line.split(",")]

        # Parse field values (second stats line)
        values_line = stats_lines[1][7:].strip()  # Remove "stats: " prefix
        field_values = [value.strip(" ") for value in values_line.split(",")]

        if len(field_names) != len(field_values):
            values = _align_field_values(field_names, field_values)
            if values is None:
                return result
            field_values = values

        # Create a dictionary mapping field names to values
        stats_dict = dict(zip(field_names, field_values))

        # Extract required values
        if "name" in stats_dict:
            result["name"] = stats_dict["name"]
        if "GFLOPs" in stats_dict:
            try:
                result["gflops"] = float(stats_dict["GFLOPs"])
            except ValueError:
                print(f"Warning: Could not convert GFLOPs value to float: {stats_dict['GFLOPs']}")
        if "timeMs" in stats_dict:
            try:
                result["time_ms"] = float(stats_dict["timeMs"])
            except ValueError:
                print(f"Warning: Could not convert timeMs value to float: {stats_dict['timeMs']}")

    # Parse error
    error_match = re.search(r"Verifies OK .* \((\d+\.\d+e[+-]\d+) < \d+\.\d+\)", output)
    if error_match:
        try:
            result["error"] = float(error_match.group(1))
        except ValueError:
            print(f"Warning: Could not convert error value to float: {error_match.group(1)}")

    return result


def _align_field_values(field_names: List[str], field_values: List[str]) -> Optional[List[str]]:
    """
    Try to align values to names by splitting any value on whitespace.
    Returns a new list of values if it matches names, otherwise None.
    """
    if len(field_names) == len(field_values):
        return field_values

    # Split each value on whitespace and re‐collect
    aligned = []
    for v in field_values:
        aligned.extend([piece for piece in v.split() if piece])

    if len(aligned) == len(field_names):
        print("Warning: Mismatched field names and values, but corrected by splitting")
        print(field_names)
        print(aligned)
        return aligned

    print("Warning: Mismatched field names and values after splitting:")
    print(field_names)
    print(aligned)
    return None


def worker_process(
    gpu_id: int,
    command_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    miopen_driver_path: str,
    iterations: int,
) -> None:
    """
    Worker process function to process commands on a specific GPU.

    Args:
        gpu_id: GPU ID to use for execution
        command_queue: Queue of commands to execute
        result_queue: Queue to store execution results
        miopen_driver_path: Path to MIOpenDriver executable
        iterations: Number of iterations to run for each command
    """
    print(f"Worker process {gpu_id} started")

    # Process commands until the queue is empty
    while True:
        try:
            # Get the next command from the queue, with a timeout
            # A timeout is used so we can check if the queue is empty
            try:
                # Now we expect a tuple of (index, command)
                cmd_index, command = command_queue.get(timeout=0.1)
            except Exception:
                # If the queue is empty, exit the loop
                if command_queue.empty():
                    break
                continue

            # Format the command according to specifications
            prefixed_cmd = f"{miopen_driver_path} {command} --iter {iterations}"

            # Add GPU selection environment variable
            full_cmd = f"ROCR_VISIBLE_DEVICES={gpu_id} {prefixed_cmd}"

            print(f"Executing on GPU {gpu_id}: {full_cmd}")

            try:
                # Execute command
                output = execute_command(full_cmd)

                # Parse output with original index
                result = parse_output(command, output, cmd_index)

                # Put result in the result queue
                result_queue.put(result)

            except Exception as e:
                print(f"Error processing command on GPU {gpu_id}: {e}")

        except Exception as e:
            print(f"Worker process {gpu_id} encountered an error: {e}")

    print(f"Worker process {gpu_id} finished")


def write_csv(results: List[Dict], output_file: str) -> None:
    """
    Write results to a CSV file.

    Args:
        results: List of command execution results (as dictionaries)
        output_file: Path to the output CSV file
    """
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Command", "Direction", "Algorithm", "Solution", "Name", "GFLOPs", "TimeMs", "Error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "Command": result["command"],
                    "Direction": result["direction"],
                    "Algorithm": result["algorithm"],
                    "Solution": result["solution"],
                    "Name": result["name"],
                    "GFLOPs": result["gflops"],
                    "TimeMs": result["time_ms"],
                    "Error": result["error"],
                }
            )


def main() -> None:
    """
    Main function to process commands and generate the CSV output.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Read input file and preserve order by adding index
    try:
        commands = []
        with open(args.input, "r") as f:
            for i, line in enumerate(f):
                if line.strip():
                    # Store as tuple of (index, command)
                    commands.append((i, line.strip()))
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Create multiprocessing queues
    command_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Fill command queue with indexed commands
    for cmd_tuple in commands:
        command_queue.put(cmd_tuple)

    # Determine number of processes to create (min of num_gpus and commands)
    num_processes = min(args.gpus, len(commands))
    print(f"Starting {num_processes} worker processes")

    # Create and start worker processes
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=worker_process, args=(i, command_queue, result_queue, args.miopendriver_path, args.iter)
        )
        process.daemon = True
        process.start()
        processes.append(process)

    # Wait for all processes to finish (with timeout)
    for process in processes:
        process.join(timeout=3600)  # 1 hour timeout
        if process.is_alive():
            print(f"Warning: Process {process.pid} is still running after timeout")

    # Collect results
    results = []
    while not result_queue.empty():
        try:
            results.append(result_queue.get(timeout=0.1))
        except Exception:
            break

    # Sort results by original index to maintain input file order
    results.sort(key=lambda x: x["index"])

    # Write results to CSV
    try:
        write_csv(results, args.output)
        print(f"Successfully processed {len(results)} commands and wrote results to {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    main()
