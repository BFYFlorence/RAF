import numpy as np
import Lacomplex as Lc
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.colors as mcolors

np.set_printoptions(suppress=True)                          # 取消科学计数显示
np.set_printoptions(threshold=np.inf)                       # 没有省略号，显示全部数组


path = "AB.pdb"           # ./frames/1ps.pdb

lc = Lc.lacomplex()

A_atom, B_atom = lc.ReadCor_CA(path)
# print(B_atom)

contact_pair = lc.CalContact(A_atom, B_atom)[1]
print(contact_pair)

# print(contact_pair)

contact_dif = lc.ConDifPerFra(contact_pair)

# print(contact_dif[0])

contact_var = lc.ConStat(contact_dif, False)
print(contact_var)


# print(contact_var)
# lc.ConSituPos_Var(contact_var)

pos = [53]
lc.ConDisTrend(contact_dif, pos)