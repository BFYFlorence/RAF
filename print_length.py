import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import re
from tensorflow import keras
import tensorflow as tf

m = tf.keras.metrics.Accuracy()
m.update_state([1, 2, 3, 4], [0, 2, 3, 4])
# Out[4]: <tf.Variable 'UnreadVariable' shape=() dtype=float32, numpy=4.0>
m.result().numpy()