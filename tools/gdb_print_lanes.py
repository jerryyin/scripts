def print_variable_for_lanes(var_name, start=1, end=32):
    """Print the value of a given variable for all lanes in a compact format."""""
    values = []  # Store values from each lane

    for i in range(start, end + 1):
        try:
            gdb.execute(f"lane {i}", to_string=True)  # Switch to thread i
            gdb.execute("frame 0", to_string=True)  # Ensure correct frame is selected
            result = gdb.execute(f"p {var_name}", to_string=True)  # Print variable
            value = result.split('=')[1].strip()  # Extract the value
        except gdb.error:
            value = "N/A"  # Handle missing variable gracefully
        values.append(value)  # Store the value

    # Print all values as a compact array-like format
    print(f"{var_name} values: [" + ", ".join(values) + "]")

# Example usage: print the 'first' variable across lanes for the range specified
# (gdb) source print_lanes.py
# (gdb) python print_variable_for_lanes("first.data")
# (gdb) python print_variable_for_lanes("lIdx", 0, 3)
