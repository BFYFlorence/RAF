import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
# from Lacomplex import Lacomplex
import re
import time
import os
# from tensorflow import keras
# import tensorflow as tf
np.set_printoptions(suppress=True)  # 取消科学计数显示
# lc = Lacomplex()
# 数据组织形式：A_Phi, A_Psi, B_Phi, B_Psi


# for i in [310.0,321.5,333.4,345.8,358.6,371.9,385.7,400.0]:
#     print("parm ../new.top\ntrajin ../remd.{0}K.mdcrd\ncluster c1 \\\n kmeans clusters 10 randompoint maxit 500 \\\n rms :*@C,N,O,CA,CB&!@H= \\\n sieve 10 random \\\n out ./repre_{0}K/cnumvtime.dat \\\n summary ./repre_{0}K/summary.dat \\\n info ./repre_{0}K/info.dat \\\n cpopvtime ./repre_{0}K/cpopvtime.agr normframe \\\n repout ./repre_{0}K/rep repfmt pdb \\\n singlerepout ./repre_{0}K/singlerep.nc singlerepfmt netcdf \\\n avgout ./repre_{0}K/avg avgfmt pdb\nrun\nclear all".format(i))
for i in range(384):
    # print(i*4, sep="", end=" ")
    print(i+1, sep="", end=" ")

