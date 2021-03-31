import time
import os
from Lacomplex import Lacomplex
import numpy as np
import pandas as pd
lc = Lacomplex()
atoms = ["N", "CA", "C"]

# cutoff = -900
# task = [""]
# shape = "cubic"
# d = 116.6782
"""if shape == "dodecahedron":
    pbc_box = [[d, 0, d * 0.5],
               [0, d, d * 0.5],
               [0, 0, d * 0.5 * np.sqrt(2)]]
if shape == "cubic":
    pbc_box = [[d, 0, 0],
               [0, d, 0],
               [0, 0, d]]"""

a_atom_cor_o, b_atom_cor_o, a_atom_nam, b_atom_nam, a_b_heavy_si = lc.readHeavyAtom("./start.pdb", monitor=True)

def calculate_dihedral(atom_cor, type=None):
    lc.dihedral_atom_order(a_atom_nam)
    lc.dihedral_atom_order(b_atom_nam)

    a_atom_cor = atom_cor[:len(a_atom_nam)]
    b_atom_cor = atom_cor[len(a_atom_nam):]

    ang_a_atom_cor = []
    ang_b_atom_cor = []

    for k in range(len(a_atom_nam)):
        if a_atom_nam[k].split("-")[2] in atoms:
            ang_a_atom_cor.append(a_atom_cor[k])

    for p in range(len(b_atom_nam)):
        if b_atom_nam[p].split("-")[2] in atoms:
            ang_b_atom_cor.append(b_atom_cor[p])

    result_a = []
    result_b = []

    if type == "Phi":
        ang_a_atom_cor.reverse()
        ang_b_atom_cor.reverse()

    ang_a_atom_cor = np.array(ang_a_atom_cor)
    ang_b_atom_cor = np.array(ang_b_atom_cor)

    ang_a_residue = []
    ang_b_residue = []

    for m in range(1, int(ang_a_atom_cor.shape[0] / 3) + 1):
        ang_a_residue.append(ang_a_atom_cor[3 * (m - 1):3 * m + 1])
    for n in range(1, int(ang_b_atom_cor.shape[0] / 3) + 1):
        ang_b_residue.append(ang_b_atom_cor[3 * (n - 1):3 * n + 1])

    for q in ang_a_residue:
        result_a.append([lc.dihedral(q[0], q[1], q[2], q[3]) if q.shape[0] == 4 else 360])
    for w in ang_b_residue:
        result_b.append([lc.dihedral(w[0], w[1], w[2], w[3]) if w.shape[0] == 4 else 360])

    if type == "Phi":
        result_a.reverse()
        result_b.reverse()

    return result_a, result_b


def process(FileName, pre_EMA, first):
    a = np.exp(-20 / 60)  # dt/τ        ########
    atom_cor = []  # 存储A、B重原子坐标
    end = len(a_b_heavy_si)

    with open(FileName, 'r') as f:
        n = 0
        for i in f.readlines():
            if a_b_heavy_si[n]:
                record = i.strip().split()
                atom_cor.append([float(j) * 10 for j in record])
            n += 1
            if n >= end:
                break

    atom_cor = np.array(atom_cor)
    # atom_cor_O = np.vstack((a_atom_cor_o, b_atom_cor_o))
    # lc.process_pbc_cubic(origin_cor=atom_cor_O, atom_cor=atom_cor, pbc=pbc_box)

    result_A_phi = []
    result_A_psi = []
    result_B_phi = []
    result_B_psi = []
    phi = calculate_dihedral(atom_cor, type="Phi")
    psi = calculate_dihedral(atom_cor, type="Psi")
    result_A_phi.append(phi[0][1:])  # 去除掉360，防止逆矩阵为奇异阵
    result_B_phi.append(phi[1][1:])
    result_A_psi.append(psi[0][:-1])  # 去除掉360，防止逆矩阵为奇异阵
    result_B_psi.append(psi[1][:-1])

    sample = np.hstack((result_A_phi, result_A_psi, result_B_phi, result_B_psi))
    sample = np.squeeze(sample, axis=2)
    # print(np.squeeze(sample, axis=2).shape)
    # 选取特征, 依靠LDA的结果来选择前几

    new_w = np.load('./dihedral_new_w.npy', allow_pickle=True)
    n_weightIndice_order = np.load('./dihedral_n_weightIndice_order.npy', allow_pickle=True)
    new_sample = np.squeeze(sample[:, n_weightIndice_order], axis=1)

    value = np.mat(new_sample) * np.mat(new_w)  # 必须是矩阵相乘，而不是array
    # print(new_sample)

    if not first:
        EMA = a * value + (1 - a) * pre_EMA
    else:
        EMA = value
        first = False
    # print("current:", value)
    # print("EMA:", EMA)
    return EMA, first, value

def MD_monitor():
    # 因为导出的pdb只有蛋白质链上会有Chain ID，所以不用担心会读取到其他的信息

    EMA = 0  # Exponential Moving Average, EMA
    first = True
    EMA_npy = []
    value_npy = []
    n = 0

    while True:
        FileName = './extract_cor.txt'
        if os.path.exists(FileName):
            # Read fime name
            original_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(FileName).st_mtime))
            time.sleep(1)
            # print file modified time
            modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(FileName).st_mtime))
            if original_time != modified_time:
                time.sleep(3)  # 避免脏读
                start_t = time.time()
                with open('./log_dih.txt', 'a') as f:
                    f.write("###############  frame:{0} ###############".format(n) + '\n')
                    f.write("Original : " + original_time + '\n')
                    f.write("Now      : " + modified_time + '\n')
                    f.write("It has been changed!" + '\n')
                    EMA, first, value = process(FileName, EMA, first)
                    EMA_npy.append(EMA)
                    value_npy.append(value)
                    np.save("./EMA_dih.npy", np.array(EMA_npy))
                    np.save("./value_dih.npy", np.array(value_npy))

                    # print(EMA)
                    f.write("value    : " + str(value) + '\n')
                    f.write("EMA      : " + str(EMA) + '\n')
                    end_t = time.time()
                    f.write("Time used: " + str(end_t - start_t) + '\n')
                    f.write("############### Done process! ###############" + '\n')
                f.close()
                n += 1

MD_monitor()

