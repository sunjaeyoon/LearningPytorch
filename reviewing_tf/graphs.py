import tensorflow as tf
import numpy as np

x = [[2., 2.], [1.,1.]]

m = tf.matmul(x,x)

print(m)