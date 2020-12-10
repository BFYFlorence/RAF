import time
import os
from Lacomplex import Lacomplex
import numpy as np
import pandas as pd
lc = Lacomplex()

def process(FileName, pre_EMA, first):
    a = 0.002 / 20 # dt/τ
    atom_cor = []  # 存储A、B重原子坐标
    end = len(a_b_heavy_si)
    with open(FileName, 'r') as f:
        n = 0
        for i in f.readlines():
            if a_b_heavy_si[n]:
                record = i.strip().split()
                atom_cor.append([float(j)*10 for j in record])
            n += 1
            if n >= end:
                break
    atom_cor = np.array(atom_cor)
    a_check_cor = atom_cor[:len(a_atom_nam)]
    b_check_cor = atom_cor[len(a_atom_nam):]
    lc.calContact(a_check_cor, b_check_cor, a_atom_nam=a_atom_nam, b_atom_nam=b_atom_nam, filename='extract_cor', save_dis=True)

    aa_contact = list(np.load('./aa_contact.npy', allow_pickle=True).item())
    aa_contact.sort()
    contact_dif = np.zeros(shape=(1, len(aa_contact)))

    dataframe = pd.read_csv('./extract_cor.csv')
    new_col = ['Unnamed: 0']
    new_index = []
    # 去除原子序数
    for col in dataframe.columns[1:]:
        record = col.split('-')
        new_col.append(record[0] + '-' + record[1] + '-' + record[2])
    for row in dataframe[dataframe.columns[0]]:
        record = row.split('-')
        new_index.append(record[0] + '-' + record[1] + '-' + record[2])
    dataframe.columns = new_col
    dataframe.index = new_index

    # 构建含有CB的cp
    for l in range(len(aa_contact)):
        cp = aa_contact[l]
        a_rec = cp[0].split('-')
        b_rec = cp[1].split('-')
        a_atom = a_rec[0] + '-' + a_rec[1] + '-' + ['CB' if a_rec[1] != 'GLY' else 'CA'][0]
        b_atom = b_rec[0] + '-' + b_rec[1] + '-' + ['CB' if b_rec[1] != 'GLY' else 'CA'][0]
        contact_dif[0][l] = dataframe[a_atom][b_atom]  # 先列后行,索引需要减2

    w = np.load('./w.npy', allow_pickle=True)
    # print(contact_dif)
    value = (np.mat(contact_dif) * w).tolist()[0][0]

    if not first:
        EMA = (1-a) * value + a * pre_EMA
    else:
        EMA = value
        first = False
    print("current:", value)
    print("EMA:", EMA)

    return EMA, first

# 因为导出的pdb只有蛋白质链上会有Chain ID，所以不用担心会读取到其他的信息
a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam, a_b_heavy_si = lc.readHeavyAtom(lc.frame_path+lc.frame_name.format(892),monitor=True)
# np.save("a_atom_nam.npy", a_atom_nam)
# np.save("b_atom_nam.npy", b_atom_nam)

EMA = 0  # Exponential Moving Average, EMA
first = True

while True:
    FileName = './extract_cor.txt'
    if os.path.exists(FileName):
        # Read fime name
        original_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(FileName).st_mtime))

        time.sleep(10)

        # print file modified time
        modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(FileName).st_mtime))
        if original_time!=modified_time:
            print("Original:", original_time)
            print("Now     :", modified_time)
            print("It has been changed!")
            EMA, first = process(FileName, EMA, first)
            print("Done process!")
            if EMA >= 200:
                task = ["",]
                for k in task:
                    os.system("qdel {0}".format(k))
