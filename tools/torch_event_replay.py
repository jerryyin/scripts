from TraceLens import EventReplayer
import pandas as pd
import ast
import sys
import traceback

# === Configuration ===
# Default list of ops to process if not specified by user
OPS_INTEREST = ['aten::addmm']

# Required columns for parsing events
REQUIRED_COLUMNS = [
    "Input Dims_first",
    "Input Strides_first",
    "Input type_first",
    "Concrete Inputs_first",
]

def validate_columns(df):
    """Ensure all required columns are present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"Missing required columns in Excel sheet:\n  {missing}")
        sys.exit(1)

def load_report(report_path, sheet_name='GEMM'):
    """Load Excel report as DataFrame."""
    try:
        df = pd.read_excel(report_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error loading report: {e}")
        sys.exit(1)
    return df

def extract_args(row):
    """Dynamically extract *_first columns as args dict."""
    args = {}
    for col in row.index:
        if col.endswith('_first'):
            key = col.replace('_first', '')
            val = row[col]
            if isinstance(val, str):
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    pass
            args[key] = val
    return args

def row_to_event(row):
    """Convert a DataFrame row to an event dict."""
    event = {
        'name': row.get('name', 'unknown'),
        'args': extract_args(row)
    }
    return event

def replay_events(df, ops_interest):
    """Replay events filtered by operation names, showing progress."""
    df_filtered = df[df['name'].isin(ops_interest)]
    total = len(df_filtered)
    if total == 0:
        print(f"No events found for ops: {ops_interest}")
        return

    print(f"Starting replay of {total} events for ops: {ops_interest}\n")

    processed_count = 0
    failed_count = 0
    for i, (_, row) in enumerate(df_filtered.iterrows(), start=1):
        event = row_to_event(row)
        try:
            replayer = EventReplayer(event)
            replayer.replay()
            processed_count += 1
        except Exception as e:
            failed_count += 1
            event_name = event.get('name', 'unknown')
            print(f"⚠️ Error on event {i}/{total} ({event_name}): {e}")
            traceback.print_exc(limit=1)  # keep it short but informative
            continue  # skip to next row

        # Print progress every 10 or at key points
        if i % 10 == 0 or i == total:
            pct = (i / total) * 100
            print(f"Progress: {i}/{total} ({pct:.1f}%)")

    print(f"\nFinished replaying {processed_count} events "
          f"(failed {failed_count}, total {total}).")

if __name__ == "__main__":
    # Example usage:
    #   python replay_from_report.py /path/to/report.xlsx aten::addmm aten::matmul
    if len(sys.argv) < 2:
        print("Usage: python replay_from_report.py <report_path> [op1 op2 ...]")
        sys.exit(1)

    report_path = sys.argv[1]
    ops_interest = sys.argv[2:] if len(sys.argv) > 2 else OPS_INTEREST

    df = load_report(report_path)
    validate_columns(df)
    replay_events(df, ops_interest)
