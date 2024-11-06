# Usage: 
# Gen rand output: 
#    python genRandInput.py input.bin --shape 1 3 244 244
# Gen readable output from bin input: 
#    python3 ~/scripts/iree/genRandInput.py readable.txt --input input.bin --shape 1 2 3

import numpy as np
import struct
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate a random tensor or convert a binary tensor file to a readable text format.")
parser.add_argument("output", type=str, help="The output file name (e.g., input.bin, readable.txt)")
parser.add_argument("--shape", type=int, nargs="+", required=True,
                                        help="The shape of the tensor for generation (e.g., 1 3 224 224). Required for generating a new file.")
parser.add_argument("--input", type=str, help="Specify a binary input file to convert to a readable text format.")
args = parser.parse_args()

if args.input:
    # Read from binary file and write to a readable text format
    with open(args.input, "rb") as f:
        # Calculate the number of elements based on the provided shape
        num_elements = np.prod(args.shape)
        
        # Read and unpack the data from the binary file
        bytearr = f.read()
        data = struct.unpack("%sf" % num_elements, bytearr)
        
        # Reshape the data to the specified shape
        tensor = np.array(data).reshape(args.shape)
        
        # Save the tensor in a readable format, preserving the shape
        with open(args.output, "w") as txt_file:
            txt_file.write(np.array2string(tensor, separator=', ', precision=6))
        print(f"Readable tensor saved to {args.output}")
else:
    # Generate a random tensor with the specified shape
    rng = np.random.default_rng(19)
    a = rng.random(args.shape).astype(np.float32)
    
    # Write the tensor to a binary file
    with open(args.output, "wb") as f:
        bytearr = struct.pack("%sf" % a.size, *a.flatten())
        f.write(bytearr)
    print(f"Binary tensor saved to {args.output}")

