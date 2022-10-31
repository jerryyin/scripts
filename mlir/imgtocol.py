# coding: utf-8
import sys
import numpy
import numpy_ml

#pattern = [0.5, -1, 0.75]
pattern = [0.5, -0.8, 0.75]
input = numpy.ndarray(shape=(128*8), dtype=float)

for i in range(0,128*8):
    input[i] = pattern[i%3]

input = input.reshape((128,1,1,8))
col = numpy_ml.neural_nets.utils.im2col(input, (1,1,8,256), 0, 1, 0)
numpy.set_printoptions(threshold=sys.maxsize)
res = numpy.transpose(col[0])
print(res)
print(res.shape)
