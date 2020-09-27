import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


np.set_printoptions(suppress=True)  # 取消科学计数显示




# logits = tf.constant([0, 5, 9, 1, 7, 1, 0, 1])

# labels = tf.ones((8, ), dtype=tf.int32)

# acc, acc_op = tf.metrics.binary_accuracy(logits, labels)

# print(acc)

m = tf.keras.metrics.BinaryAccuracy()

m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
print(m.result().numpy())

