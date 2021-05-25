import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
# from Lacomplex import Lacomplex
import re
import time
import os
# from tensorflow import keras
import tensorflow as tf
np.set_printoptions(suppress=True)  # 取消科学计数显示
# lc = Lacomplex()
# 数据组织形式：A_Phi, A_Psi, B_Phi, B_Psi

# x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
def test(x):
    return x+1

a = test

print(a(1))
