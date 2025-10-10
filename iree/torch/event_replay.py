from TraceLens import TreePerfAnalyzer, EventReplayer
import pandas as pd
import ast
# read sheet from excel

perf_report_path = "/zyin/h200-2025-07-16.xlsx"
df_unique_ops = pd.read_excel(perf_report_path, sheet_name='GEMM')

def row_to_evt(row):
    event = {
        'name': row['name'],
        'args': {
            'Input Dims': ast.literal_eval(row['Input Dims_first']),
            'Input Strides': ast.literal_eval(row['Input Strides_first']),
            'Input type': ast.literal_eval(row['Input type_first']),
            'Concrete Inputs': ast.literal_eval(row['Concrete Inputs_first']),
        }
    }
    return event

repro_data_list = []
processed_count = 0
# lets say we are interested in the following ops
ops_interest = ['aten::addmm']

df_ops_interest = df_unique_ops[df_unique_ops['name'].isin(ops_interest)].copy()

for index, row in df_ops_interest.iterrows():
    event = row_to_evt(row)
    # Initialize EventReplayer similar to above
    #replayer = EventReplayer(event, lazy=True, verbose=False)
    replayer = EventReplayer(event, device="cuda:0", verbose=False)
    # Extract the serializable info
    repro_info = replayer.get_repro_info()
    print(repro_info)
    replayer.replay()
    repro_data_list.append(repro_info)
    processed_count += 1
print(f"Processed {processed_count} events.")
