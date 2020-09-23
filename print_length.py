import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import re
from tensorflow import keras
import tensorflow as tf

# print(train1[1499])
# print(train2[1499])

model = tf.saved_model.load("./models/model_20")


