import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from Lacomplex import Lacomplex
import re
import time
import os
# from tensorflow import keras
# import tensorflow as tf
np.set_printoptions(suppress=True)  # 取消科学计数显示
lc = Lacomplex()
# 数据组织形式：A_Phi, A_Psi, B_Phi, B_Psi
# bind
bind_A_Phi = [38,  39,  40, 100, 102, 105, 107, 156, 170, 171, 172, 173, 175, 213, 215, 251, 260, 264]
bind_A_Psi = [10,  30,  39,  40,  41,  88, 102, 105, 107, 125, 148, 156, 170, 171, 173, 174, 175, 176, 177, 216, 251, 253, 260, 264]
bind_B_Phi = [6,   7,  35,  41,  58,  63,  99, 102, 155, 156, 171, 172, 173, 175, 211, 229, 252]
bind_B_Psi = [6,   8,  40,  41,  63,  67,  88,  94,  99, 100, 102, 108, 117, 125, 149, 151, 156, 157, 173, 174, 177, 211, 229, 264]

# nobind
nobind_A_Phi = [5,   7,   8,  10,  36,  37,  38,  39,  40,  58,  62,  63, 128, 155, 156, 171, 172, 173, 174, 175, 213, 215]
nobind_A_Psi = [7,  10,  37,  40,  41,  42,  63,  88, 100, 125, 155, 156, 169, 170, 171, 173, 174, 175, 176, 177, 184, 215, 250, 264]
nobind_B_Phi = [37,  41,  52,  55,  56,  57,  58,  78, 155, 156, 170, 171, 172, 173, 175, 247, 248, 250, 251, 252]
nobind_B_Psi = [7,  40,  52,  56,  67,  78, 79,  93, 125, 156, 157, 171, 173, 174, 175, 176, 177, 210, 211, 226, 249, 250, 251, 253]



# print(position)
# 340
dataframe = pd.read_csv('./{0}.csv'.format("extract_cor"))
print(dataframe)