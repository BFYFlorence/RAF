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

# crystal_WT_phi = []
crystal_WT_psi = []
cry_repacking_phi = []
cry_repacking_psi = []
serial = 5
num = 10
target = "phi"
temperature = "300K"

# 残基序号
Strand = [[4, 5, 6, 7, 8], [15, 16, 17, 18], [43, 44, 45, 46, 47], [72, 73, 74, 75, 76, 77]]
Alphahelix = [[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], [65, 66, 67, 68]]
WT_seq = ['THR', 'SER', 'ASN', 'THR', 'ILE', 'ARG', 'VAL', 'PHE', 'LEU', 'PRO', 'ASN', 'LYS', 'GLN', 'ARG',
          'THR', 'VAL', 'VAL', 'ASN', 'VAL', 'ARG', 'ASN', 'GLY', 'MET', 'SER', 'LEU', 'HIS', 'ASP', 'CYS',
          'LEU', 'MET', 'LYS', 'ALA', 'LEU', 'LYS', 'VAL', 'ARG', 'GLY', 'LEU', 'GLN', 'PRO', 'GLU', 'CYS',
          'CYS', 'ALA', 'VAL', 'PHE', 'ARG', 'LEU', 'LEU', 'HIS', 'GLU', 'HIS', 'LYS', 'GLY', 'LYS', 'LYS',
          'ALA', 'ARG', 'LEU', 'ASP', 'TRP', 'ASN', 'THR', 'ASP', 'ALA', 'ALA', 'SER', 'LEU', 'ILE', 'GLY',
          'GLU', 'GLU', 'LEU', 'GLN', 'VAL', 'ASP', 'PHE', 'LEU']

repacking_seq = ['ALA', 'ASP', 'ARG', 'THR', 'ILE', 'GLU', 'VAL', 'GLU', 'LEU', 'PRO', 'ASN', 'LYS', 'GLN',
                 'ARG', 'THR', 'VAL', 'ILE', 'ASN', 'VAL', 'ARG', 'PRO', 'GLY', 'LEU', 'THR', 'LEU', 'LYS',
                 'GLU', 'ALA', 'LEU', 'LYS', 'LYS', 'ALA', 'LEU', 'LYS', 'VAL', 'ARG', 'GLY', 'ILE', 'ASP',
                 'PRO', 'ASN', 'LYS', 'VAL', 'GLN', 'VAL', 'TYR', 'LEU', 'LEU', 'LEU', 'SER', 'GLY', 'ASP',
                 'ASP', 'GLY', 'ALA', 'GLU', 'GLN', 'PRO', 'LEU', 'SER', 'LEU', 'ASN', 'HIS', 'PRO', 'ALA',
                 'GLU', 'ARG', 'LEU', 'ILE', 'GLY', 'LYS', 'LYS', 'LEU', 'LYS', 'VAL', 'VAL', 'PRO', 'LEU']
# for k in range(1, serial + 1):
k = 1
print("k:", k)
crystal_WT_phi = []
for i in range(1, num + 1):
    print("i:", i)
    sample = np.load("/Users/erik/Desktop/RAF/crystal_WT/{2}/{1}/phi_{0}.npy".format(i, k, temperature),
                     allow_pickle=True).tolist()
    crystal_WT_phi += sample

crystal_WT_phi = np.array(crystal_WT_phi)
print(crystal_WT_phi[:,2].shape)
