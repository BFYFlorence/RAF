import numpy as np
# 数据组织形式：A_Phi, A_Psi, B_Phi, B_Psi

# ARG = []
with open("./ARG.dat", 'r') as file:
    for i in file.readlines():
        record = i.split(",")
        print(len(record))

