#!/usr/bin/env python3

import os
import sys
import re
import json


KEY_TRACE_EVENTS = "traceEvents"
KEY_CATEGORY = "cat"
KEY_ARGS = "args"
KEY_NAME = "name"
KEY_PID = "pid"
KEY_OP = "op"


VALUE_PROCESS_NAME = "process_name"
VALUE_CATEGORY_DATAFLOW = "DataFlow"
VALUE_CATEGORY_OP = "Op"

def filter_trace_events(trace_events):

    trace_event_array = trace_events[KEY_TRACE_EVENTS]

    filtered_trace_event_array = []
    for event in trace_event_array:

        if event[KEY_NAME] == VALUE_PROCESS_NAME:
            filtered_trace_event_array.append(event)
            continue

        # if event[KEY_CATEGORY] == VALUE_DATAFLOW:
        #     continue

        if event[KEY_CATEGORY] != VALUE_CATEGORY_OP:
            continue

        # node_name = event[KEY_ARGS][KEY_NAME]
        # if re.search(r'encoder/layer_2/', node_name):
        #     filtered_trace_event_array.append(event)

        op_name = event[KEY_ARGS][KEY_OP]
        if re.search(r'Conv2DBackpropFilter', op_name):
            filtered_trace_event_array.append(event)
            
    filtered_trace_events = { KEY_TRACE_EVENTS : filtered_trace_event_array }
    return filtered_trace_events




def edit_timeline(input_file, output_file):

    with open(input_file) as rf:
        trace_data = rf.read()

    trace_events = json.loads(trace_data)
    filtered_trace_events = filter_trace_events(trace_events)
        
    with open(output_file, "w") as wf:
        wf.write(json.dumps(filtered_trace_events))


if __name__ == '__main__':

    input_file = "resnet50_trace.json"
    output_file = "conv2dbackpropfilter-resnet50_trace.json"
    
    edit_timeline(input_file, output_file)
