#!/usr/bin/env python3

import os
import sys
import re
import json


KEY_TRACE_EVENTS = "traceEvents"
KEY_ARGS = "args"
KEY_NAME = "name"
KEY_PID = "pid"
VALUE_PROCESS_NAME = "process_name"


def process_trace_events(name_mapping, trace_events, used_pids):

    trace_event_array = trace_events[KEY_TRACE_EVENTS]

    pid_map = {}

    # generate the pid_map 
    for event in trace_event_array:
        if event[KEY_NAME] != VALUE_PROCESS_NAME:
            continue
        pid = event[KEY_PID]
        new_pid = pid
        if pid in used_pids:
            new_pid = max(used_pids) + 1
        pid_map[pid] = new_pid
        used_pids.add(new_pid)

    # use the pid_map to update the pids
    for event in trace_event_array:
        pid = event[KEY_PID]
        event[KEY_PID] = pid_map[pid]

    # update the args[name] with the name_mapping
    for event in trace_event_array:
        if event[KEY_NAME] != VALUE_PROCESS_NAME:
            continue
        args_name = event[KEY_ARGS][KEY_NAME]
        for orig, new in name_mapping.items():
            args_name = args_name.replace(orig, new)
        event[KEY_ARGS][KEY_NAME] = args_name


    def keep_this_process(process_name):
        pattern = r'<Kernel> Compute'
        if re.search(pattern, process_name):
            return True
        return False
        
    pids_to_keep = set()
    for event in trace_event_array:
        if event[KEY_NAME] != VALUE_PROCESS_NAME:
            continue
        process_name = event[KEY_ARGS][KEY_NAME]
        if keep_this_process(process_name):
            pids_to_keep.add(event[KEY_PID])

    filtered_trace_event_array = []
    for event in trace_event_array:
        if event[KEY_PID] in pids_to_keep:
            filtered_trace_event_array.append(event)

    filtered_trace_events = { KEY_TRACE_EVENTS : filtered_trace_event_array }
    return filtered_trace_events, used_pids




def stitch_timelines(dirs_and_name_mappings, input_file, output_file):

    merged_trace_events = { KEY_TRACE_EVENTS : []}
    used_pids = set()
    
    for input_dir, name_mapping in dirs_and_name_mappings.items():

        with open(os.path.join(input_dir, input_file)) as rf:
            trace_data = rf.read()

        trace_events = json.loads(trace_data)
        
        trace_events, used_pids = process_trace_events(name_mapping, trace_events, used_pids)
        
        merged_trace_events[KEY_TRACE_EVENTS].extend(trace_events[KEY_TRACE_EVENTS])

    with open(output_file, "w") as wf:
        wf.write(json.dumps(merged_trace_events))


if __name__ == '__main__':

    dirs_and_name_mappings = {
        "GPU_0" : { "GPU:0" : "GPU:0", "CPU:0" : "CPU:0"} ,
        "GPU_1" : { "GPU:0" : "GPU:1", "CPU:0" : "CPU:1"} ,
        "GPU_2" : { "GPU:0" : "GPU:2", "CPU:0" : "CPU:2"} ,
        "GPU_3" : { "GPU:0" : "GPU:3", "CPU:0" : "CPU:3"} ,
        "GPU_4" : { "GPU:0" : "GPU:4", "CPU:0" : "CPU:4"} ,
        "GPU_5" : { "GPU:0" : "GPU:5", "CPU:0" : "CPU:5"} ,
        "GPU_6" : { "GPU:0" : "GPU:6", "CPU:0" : "CPU:6"} ,
        "GPU_7" : { "GPU:0" : "GPU:7", "CPU:0" : "CPU:7"} ,
    }

    input_file = "timeline-21.json"
    output_file = "merged-timeline-21-json"
    
    stitch_timelines(dirs_and_name_mappings, input_file, output_file)
