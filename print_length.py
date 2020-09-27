import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import re
from tensorflow import keras
import tensorflow as tf

np.set_printoptions(suppress=True)  # 取消科学计数显示

bind = np.load('./Bind.npy', allow_pickle=True)
bind = bind[500:]
# print(bind.shape)

nobind = np.load('./Nobind.npy', allow_pickle=True)
nobind = nobind[500:]
# print(nobind.shape)

bind_var = np.cov(m=bind, rowvar=False)
nobind_var = np.cov(m=nobind, rowvar=False)
# print(bind)
# print(bind.shape)

# print(np.var(bind[:, 0], ddof=1))  # N-ddof:N-1

print(bind_var)
# print(bind_var.shape)
# bind_df = pd.DataFrame(bind_var)
np.savetxt('./bind_var.csv', bind_var, delimiter=',', fmt="%.5f")
