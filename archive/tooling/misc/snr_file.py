#! /usr/bin/python3

import fileinput
import re

filenames = """
./tensorflow/core/platform/default/notification.h
./tensorflow/core/lib/io/block_builder.cc
./tensorflow/core/lib/io/table_builder.cc
./tensorflow/core/lib/core/arena.h
./tensorflow/core/lib/core/arena.cc
./tensorflow/core/lib/core/stringpiece.h
./tensorflow/core/lib/strings/ordered_code.cc
./tensorflow/core/lib/gtl/optional.h
./tensorflow/core/lib/random/weighted_picker.h
./tensorflow/core/kernels/sparse_cross_op.cc
./tensorflow/core/kernels/check_numerics_op_gpu.cu.cc
./tensorflow/core/kernels/multinomial_op_gpu.cu.cc
./tensorflow/core/kernels/parameterized_truncated_normal_op_gpu.cu.cc
./tensorflow/core/kernels/random_op_gpu.cu.cc
./tensorflow/stream_executor/lib/human_readable.h
./tensorflow/stream_executor/rocm/rocm_blas.cc
./tensorflow/contrib/layers/kernels/sparse_feature_cross_kernel.cc
"""

# with fileinput.input(filenames.split(), inplace=True) as f:
#     for line in f:
#         print (line, end='')
#         if re.search("#include <assert.h>", line):
#             print("")
#             print("// temporarily disable asserts on the ROCm platform")
#             print("#ifdef __HIP_DEVICE_COMPILE__")
#             print("  #undef assert")
#             print("  #define assert(x)")
#             print("#endif")
#             print("")


with fileinput.input(filenames.split(), inplace=True) as f:
    for line in f:
        if re.search("#include <assert.h>", line):
            print("")
            print("// temporarily disable asserts on the ROCm platform")
            print("#undef assert")
            print("#define assert(x)")
            print("")
        else :
            print (line, end='')
