# 归一化最好使用方差归一化
# 续点不需要重新平衡
# 目前还只是探索，后续要自动执行，放入超算里
# 小红师姐的课题，要跑长一点，然后观察起伏，如果只是波动就没必要求平均，直接拿出来看就好了
# 隐式溶剂模型使用amber
# 用二面角来对这两种构象进行表征，看是否正交；
# PCA验证变量间的独立性，
# 试着使用NN来区分第二维，看是否有能力区分能力
# 两种状态的原子顺序有细微的差异
# 小红师姐的课题，要先统计SASA的分布？然后再跑REMD，看均值和方差
# fit会因为算法原因让一段结构倾向于稳定在原地，这就会导致也许其余结构变化并不大，但是fit之后导致较高的rmsf
# 检查一下方差计算是否有误
# 将relative_SASA分为三个部分，小于0.36，大于0.6，介于两者之间，氨基酸的个数
# 对于副本交换的数据，追踪一段轨迹内遍历的温度是否均匀；还可以把同一个温度下的轨迹收集起来做聚类，与实验结果比较
# 对于我的课题，可以去看一下两个局部极小的dih局部特征改变的都有哪些，把dih的分布画出来
# 把REMD的轨迹求平均，大概就算跨了温度也无所谓
# 温度副本交换MD一般来说低温度的要密集一些，高温的要稀疏一些
# 对于一个副本经历的温度，看一下它的结构覆盖是否齐全，若是齐全的话就没必要那么高的温度，检查其收敛性，有没有一直保持在一个温度
# 还可以检查一下回旋半径和二级结构的数量，监测其松散程度
# 检查一下计算的效率，可能不需要那么多的核数
# 质心的约束
# 小红师姐的课题，对于为什么出现模拟崩溃的现象，有一种可能的原因是恒温器的问题，再进一步解释是质心运动，但是我现在还没搞懂速度缩放和温度控制之间的关系


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, optimizers
from matplotlib.pyplot import MultipleLocator
import os
import __main__
__main__.pymol_argv = ['pymol', '-qc']
import pymol as pm

np.set_printoptions(suppress=True)  # 取消科学计数显示
np.set_printoptions(threshold=np.inf)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Macos需要设为true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 自定义计算acc
"""def Myaccc(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1, output_type=tf.int32),
                                               tf.argmax(y_true, axis=1, output_type=tf.int32)), tf.float32)) # 保存行数
    return accuracy
"""

"""class Myacc(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(tf.argmax(y_true, axis=1, output_type=tf.int32),
                                  tf.argmax(y_pred, axis=1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0)
        self.count.assign(0)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_myacc") > 0.95 and logs.get("loss") < 0.1:
            print("\n meet requirements so cancelling training!")
            self.model.stop_training = True
"""

class Lacomplex:
    def __init__(self):
        self.contact_dis = 4.5  # 重原子之间的contact距离
        self.startFrame = 1  # 首帧
        self.endFrame = 5000 + 1  # 末帧
        self.set_name = 7  # 字典存储名称, 用于set和freq的存储
        self.aa = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "ASP", "ASN",
                   "GLU", "LYS", "GLN", "MET", "SER", "THR", "CYS", "PRO", "HIS", "ARG",
                   "HID", "ASN", "ASH", "HIE", "HIP"]

        self.data_name = ""
        self.csv_path = ""  # 表格读取路径
        self.frame_path = ""  # 存储每一帧的文件夹
        self.ANN = ""
        self.output = ""  # 分析数据输出文件夹

        self.startSet = 1  # 字典起始
        self.endSet = 10 + 1  # 字典终止

        self.Interval = 1  # 取帧间隔,看帧的名称

        self.frame_name = "md{0}.pdb"  # 每一帧的名称
        self.csv_name = "{0}.csv"  # 每一张表的名称

        self.vmd_rmsd_path = "/Users/erik/Desktop/MD_WWN/test_100ns/"
        self.rmsd_name = "trajrmsd.dat"
        self.sasa_max = {"GLY": 87.6,
                         "ALA": 111.9,
                         "VAL": 154.4,
                         "LEU": 183.1,
                         "ILE": 179.1,
                         "PHE": 206.4,
                         "TRP": 239.0,
                         "TYR": 224.6,
                         "ASP": 169.4,
                         "ASN": 172.5,
                         "GLU": 206.0,
                         "LYS": 212.5,
                         "GLN": 204.4,
                         "MET": 204.3,
                         "SER": 132.9,
                         "THR": 154.1,
                         "CYS": 139.3,
                         "PRO": 148.898,  # No hydrogen bonds found, so this figure is calculated by pymol
                         "HIS": 188.5,
                         "ARG": 249.0
                        }  # Angstroms^2, using 5-aa stride

        # self.hydrophobic_index = [[16, 20, 34, 38, 50, 53], [15, 16, 18, 20]]
        # self.hydrophobic_index = [[16, 20, 34, 38, 50, 53], [82, 83, 85, 87]]
        self.hydrophobic_index = [16, 20, 34, 38, 50, 53, 82, 83, 85, 87]  # sasa_statistics
        self.either_index = [37, 41, 45]

        self.hydrophilic_index = [36]

    def processIon(self, aa):  # 处理质子化条件
        if aa in ['ASH', 'ASN']:
            return 'ASP'
        if aa in ['HIE', 'HID', 'HIP']:
            return 'HIS'
        return aa

    def norm(self, data):  # 最好应该是方差归一化
        # min-max标准化
        min_val = np.min(data)
        max_val = np.max(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - min_val) / (max_val - min_val)
        return data

    def mergeSet(self):
        total_contact = set()
        # 遍历子集合，并取并集
        for i in range(self.startSet, self.endSet):
            path = self.csv_path + "{0}.npy".format(i)
            contact = np.load(path, allow_pickle=True).item()
            total_contact = total_contact | contact
        np.save(self.output + "total_contact_{0}.npy".format(self.data_name), total_contact)

    def check_diff(self, extract_txt, target_pdb):
        # 因为导出的pdb只有蛋白质链上会有Chain ID，所以不用担心会读取到其他的信息
        a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam, a_b_heavy_si = self.readHeavyAtom(
            self.frame_path + self.frame_name.format(4846), monitor=True)

        atom_cor = []  # 存储A、B重原子坐标
        end = len(a_b_heavy_si)
        with open(extract_txt, 'r') as f:
            n = 0
            for i in f.readlines():
                if a_b_heavy_si[n]:
                    record = i.strip().split()
                    atom_cor.append([float(j) * 10 for j in record])
                n += 1
                if n >= end:
                    break
        arr1 = np.array(atom_cor)[:len(a_atom_nam)]
        arr2 = self.readHeavyAtom(target_pdb)[0]

        print(arr1.shape)
        print(arr2.shape)
        for i in range(arr1.shape[0]):
            for j in range(3):
                if np.abs(arr1[i][j] - arr2[i][j]) >= 0.0001:
                    print("there is a difference!")
                    print(arr1[i][j], arr2[i][j])

    def readHeavyAtom(self, path, monitor=False) -> np.array:
        # 读取每条链重原子的坐标，必须要有chainID的信息
        """[[-21.368 108.599   3.145]
            [-19.74  109.906   6.386]
            [-19.151 113.618   6.922]
            [-16.405 114.786   4.541]
            ...
            [  8.717  80.336  46.425]
            [  7.828  76.961  48.018]
            [  8.38   74.326  45.331]
            [ 12.103  74.061  46.05 ]]"""
        print("Reading:", path)
        print("Better check out the last column in the input file!")
        a_atom_cor = []  # 存储A链原子坐标
        b_atom_cor = []  # 存储B链原子坐标
        a_atom_nam = []  # 存储A链原子名称
        b_atom_nam = []  # 存储B链原子名称
        a_heavy_si = []  # 存储A链对应位置是否为重原子
        b_heavy_si = []  # 存储B链对应位置是否为重原子

        index = 0  # 向atom中添加A,B两条链坐标
        chain = ""
        with open(path, 'r') as f:
            for i in f.readlines():
                record = i.strip()
                # if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                #     break

                atom = record[:4].strip()
                if atom != "ATOM":  # 检测ATOM起始行
                    continue

                serial = record[6:11].strip()  # 697
                atname = record[12:16].strip()  # CA
                resName = self.processIon(record[17:20].strip())  # PRO, 已处理过质子化条件
                if resName not in self.aa:
                    continue
                current_chain = record[21].strip()  # 获取chain_ID,A
                resSeq = record[22:26].strip()  # 3
                cor_x = record[30:38].strip()  # Å
                cor_y = record[38:46].strip()
                cor_z = record[46:54].strip()
                element = record[13].strip()  # C
                if atom == "TER" or (current_chain != chain and index):  # 检测chain_ID的变化
                    # 轨迹文件导出时没有TER行，所以由
                    # or后面条件辨别
                    index = 0

                xyz = [float(cor_x), float(cor_y), float(cor_z)]
                # eg: 2-LYS-N-697
                name = resSeq + "-" + resName + "-" + atname + "-" + serial
                if monitor:
                    # 如启用监测，保存重原子信息
                    if current_chain == "A":
                        index += 1
                        chain = current_chain
                        if element != "H":
                            a_atom_cor.append(xyz)
                            a_atom_nam.append(name)
                            a_heavy_si.append(True)
                        else:
                            a_heavy_si.append(False)
                    if current_chain == "B":
                        index += 1
                        chain = current_chain
                        if element != "H":
                            b_atom_cor.append(xyz)
                            b_atom_nam.append(name)
                            b_heavy_si.append(True)
                        else:
                            b_heavy_si.append(False)
                else:
                    if element != "H" and current_chain == "A":
                        a_atom_cor.append(xyz)
                        a_atom_nam.append(name)
                        index += 1
                        chain = current_chain
                    if element != "H" and current_chain == "B":
                        b_atom_cor.append(xyz)
                        b_atom_nam.append(name)
                        index += 1
                        chain = current_chain

        if monitor:
            return np.array(a_atom_cor), np.array(b_atom_cor), a_atom_nam, b_atom_nam, a_heavy_si + b_heavy_si
        else:
            return np.array(a_atom_cor), np.array(b_atom_cor), a_atom_nam, b_atom_nam

    def calContact(self, a_atom_cor, b_atom_cor, a_atom_nam=None, b_atom_nam=None, filename=None, save_dis=None):
        # 计算a_atom, b_atom中每个原子之间的距离矩阵,
        """             a
           [[29.82359244 30.74727287 28.20280617 ... 27.66411036]
            [30.76806536 31.93978661 29.39017023 ... 29.17042444]
        b   [28.22935699 29.43069216 26.80743378 ... 26.78451616]
            ...          ...         ...             ...
            [27.910641   29.4440505  26.99579821 ... 27.44559085]]"""
        # 初始化距离助阵
        dis_array = np.zeros(shape=(b_atom_cor.shape[0], a_atom_cor.shape[0]))
        # 初始化contact集合
        contact_pair = set()  # contact集合
        dis_array = self.euclidean(b_atom_cor, a_atom_cor)

        for a in range(a_atom_cor.shape[0]):
            for b in range(b_atom_cor.shape[0]):
                # 取小于等于4.5Å的contact
                if dis_array[b][a] <= self.contact_dis:
                    # 原子序号，先A链后B链
                    contact_pair.add((a_atom_nam[a], b_atom_nam[b]))

        # 创建数据框
        # print("dis_array.shape:", dis_array.shape)
        data_df = pd.DataFrame(dis_array)
        # print("data_df.columns:", len(data_df.columns))
        # 添加列标题
        data_df.columns = a_atom_nam
        # 添加索引
        data_df.index = b_atom_nam

        if save_dis:
            path = self.csv_path + self.csv_name.format(filename)
            print("Saving:", path)
            data_df.to_csv(path, float_format="%.5f")

        return contact_pair, data_df

    def euclidean(self, a_matrix, b_matrix):
        # 采用矩阵运算实现欧氏距离的计算
        """                     b
           [[2.23606798  1.          5.47722558]
        a   [7.07106781  4.69041576  3.        ]
            [12.20655562 9.8488578   6.4807407 ]]"""

        d1 = -2 * np.dot(a_matrix, b_matrix.T)
        d2 = np.sum(np.square(a_matrix), axis=1, keepdims=True)
        d3 = np.sum(np.square(b_matrix), axis=1)
        dist = np.sqrt(d1 + d2 + d3)

        return dist

    def dihedral(self, p1, p2, p3, p4):  # 计算单一二面角
        p12 = p2 - p1
        p13 = p3 - p1
        p42 = p2 - p4
        p43 = p3 - p4
        nv1 = np.cross(p13, p12)
        nv2 = np.cross(p43, p42)
        if nv2.dot(p12) >= 0:
            signature = -1
        else:
            signature = 1
        return (np.arccos(np.dot(nv1, nv2)/(np.linalg.norm(nv1)*np.linalg.norm(nv2)))/np.pi)*180*signature

    def dihedral_atom_order(self, atom_nam):
        n = 0
        atoms = ["N", "CA", "C"]
        for i in atom_nam:
            if i.split("-")[2] in atoms:
                if atoms[n % 3] != i.split("-")[2]:
                    raise Exception("二面角原子顺序错误")
                n += 1

    def gather_dihedral_atom(self, path, type=None):
        result_a = []
        result_b = []
        atoms = ["CA", "N", "C"]
        a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = self.readHeavyAtom(path, monitor=False)
        ang_a_atom_cor = []
        ang_b_atom_cor = []

        self.dihedral_atom_order(a_atom_nam)
        self.dihedral_atom_order(b_atom_nam)

        for k in range(len(a_atom_nam)):
            if a_atom_nam[k].split("-")[2] in atoms:
                ang_a_atom_cor.append(a_atom_cor[k])

        for p in range(len(b_atom_nam)):
            if b_atom_nam[p].split("-")[2] in atoms:
                ang_b_atom_cor.append(b_atom_cor[p])

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
            result_a.append([self.dihedral(q[0], q[1], q[2], q[3]) if q.shape[0] == 4 else 360])
        for w in ang_b_residue:
            result_b.append([self.dihedral(w[0], w[1], w[2], w[3]) if w.shape[0] == 4 else 360])

        if type == "Phi":
            result_a.reverse()
            result_b.reverse()

        return result_a, result_b  # 0代表A链，1代表B链

    def batchFrame2Dih(self):
        # 行：每一帧
        # 列：每一个二面角
        result_A_phi = []
        result_A_psi = []
        result_B_phi = []
        result_B_psi = []
        for i in range(self.startFrame, self.endFrame):
            pdb_path = self.frame_path + "/md{0}.pdb".format(i)
            phi = self.gather_dihedral_atom(pdb_path, type="Phi")
            result_A_phi.append(phi[0][1:])   # 去除掉360，防止逆矩阵为奇异阵
            result_B_phi.append(phi[1][1:])

            psi = self.gather_dihedral_atom(pdb_path, type="Psi")
            result_A_psi.append(psi[0][:-1])  # 去除掉360，防止逆矩阵为奇异阵
            result_B_psi.append(psi[1][:-1])

        np.save("result_A_phi.npy", result_A_phi)
        np.save("result_B_phi.npy", result_B_phi)
        np.save("result_A_psi.npy", result_A_psi)
        np.save("result_B_psi.npy", result_B_psi)

    def batchFrame2Dis(self):
        # 创建contact集合
        total_contact = set()
        # 遍历所有帧
        for i in range(self.startFrame, self.endFrame):
            path = self.frame_path + self.frame_name.format(i * self.Interval)
            # 读取坐标
            a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = self.readHeavyAtom(path)
            # 汇集每个batch的contact集合
            contact_pair = self.calContact(a_atom_cor, b_atom_cor, a_atom_nam=a_atom_nam,
                                           b_atom_nam=b_atom_nam,
                                           filename=i * self.Interval, save_dis=True)[0]
            total_contact = total_contact | contact_pair

        # 保存计算结果
        np.save(self.csv_path + "{0}.npy".format(self.set_name), total_contact)

    def mergebothContact(self):
        bind = np.load(self.output + "aa_contact_{0}.npy".format("NoIptg_Bind"), allow_pickle=True).item()
        nobind = np.load(self.output + "aa_contact_{0}.npy".format("NoIptg_NoBind"), allow_pickle=True).item()

        np.save(self.output + "aa_contact.npy", bind | nobind)

    def ConDifPerFra(self, save=None):
        #         列：原子对
        # 行：帧数
        # 读取汇总的所有帧的contact集合，两态
        contact_pair = np.load(self.output + "whole_contact.npy", allow_pickle=True).item()
        contact_pair = list(contact_pair)
        # 进行排序，保证contact_pair的一致性
        contact_pair.sort()
        print("contact_pair:  ", contact_pair)
        # 初始化contact距离矩阵
        contact_dif = np.zeros(shape=(self.endFrame - self.startFrame, len(contact_pair)))
        # 读取每一帧对应的原子距离csv
        for i in range(self.startFrame, self.endFrame):
            # 生成对应csv路径
            path = self.csv_path + self.csv_name.format(i * self.Interval)
            data_df = pd.read_csv(path)
            # 为索引赋值，保证随机读取
            data_df.index = data_df[data_df.columns[0]]
            for j in range(len(contact_pair)):
                contact_dif[i - self.startFrame][j] = data_df[contact_pair[j][0]][contact_pair[j][1]]  # 先列再行

        if save:
            data_df = pd.DataFrame(contact_dif)
            data_df.columns = contact_pair
            data_df.to_csv(self.output + "contact_dif_{0}.csv".format(self.data_name), float_format="%.5f")

    def ConDifPerFra_CB(self):
        # 读取aa_contact并排序
        aa_contact = list(np.load(self.output + 'aa_contact.npy', allow_pickle=True).item())
        aa_contact.sort()
        # 初始化矩阵
        contact_dif = np.zeros(shape=(self.endFrame - self.startFrame, len(aa_contact)))
        # 遍历
        for i in range(self.startFrame, self.endFrame):
            path = self.csv_path + self.csv_name.format(i * self.Interval)
            print("reading:", path)
            dataframe = pd.read_csv(path)
            # 因为原始数据框是含有原子信息的，所以要去除，只保留残基
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
            for l in range(len(aa_contact)):  # for this step, you need to sort the set
                cp = aa_contact[l]
                a_index = cp[0] + '-' + ['CB' if cp[0][-3:] != 'GLY' else 'CA'][0]
                b_index = cp[1] + '-' + ['CB' if cp[1][-3:] != 'GLY' else 'CA'][0]
                # 先列后行
                contact_dif[i - self.startFrame][l] = dataframe[a_index][b_index]

        data_df = pd.DataFrame(contact_dif)
        data_df.columns = aa_contact
        data_df.to_csv(self.output + 'contact_dif_CB_{0}.csv'.format(self.data_name), float_format="%.5f")

    def avedis(self):
        path = self.output + "contact_dif_CB_{0}.csv".format(self.data_name)  # contact_dif
        contact_dif = pd.read_csv(path)
        # 初始化均值、方差矩阵
        ave_var_array = np.zeros(shape=(len(contact_dif.columns) - 1, 2))
        # 计算每个contact的均值及方差
        for i in range(1, len(contact_dif.columns)):
            ave_var_array[i - 1][0] = np.mean(np.array(contact_dif[contact_dif.columns[i]]))
            ave_var_array[i - 1][1] = np.var(np.array(contact_dif[contact_dif.columns[i]]))

        ave_df = pd.DataFrame(ave_var_array)
        ave_df.columns = ["ave", "var"]
        ave_df.index = contact_dif.columns[1:]
        #
        ave_df.to_csv(self.output + "ave_var_dif_{0}.csv".format(self.data_name), float_format="%.5f")

    def numperaa(self):  # average_dif
        path = self.csv_path + "whole_cp.npy"
        cp = list(np.load(path, allow_pickle=True).item())
        print(len(cp))
        freq_array = [0] * (329 - 62 + 1)
        for i in cp:
            record = i[1].split('-')  # 0，1代表画A链还是B链
            pos = int(record[0]) - 62
            freq_array[pos] = freq_array[pos] + 1

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('pos', fontsize=20)
        ax1.set_ylabel("freq", fontsize=20)
        ax1.plot(range(62, 330), freq_array)  # ###############!!!!!!!
        plt.show()

    def aveDistribution(self):  # average_dif
        bind_path = self.csv_path + self.csv_name.format("ave_var_dif_bind")
        nobind_path = self.csv_path + self.csv_name.format("ave_var_dif_nobind")

        bind = pd.read_csv(bind_path)
        nobind = pd.read_csv(nobind_path)
        print(len(bind.index))

        bind.index = bind[bind.columns[0]]
        nobind.index = nobind[nobind.columns[0]]

        bind.drop(['Unnamed: 0'], axis=1, inplace=True)  # inplace就地修改，1指定列
        nobind.drop(['Unnamed: 0'], axis=1, inplace=True)  # inplace就地修改，1指定列

        bind = bind.values
        nobind = nobind.values
        # print(nobind.index)
        # print(bind.index)
        # x为bind，y为nobind
        ave_x = []
        ave_y = []
        var_x = []
        var_y = []

        for i in range(bind.shape[0]):
            ave_x.append(bind[i][0])
            var_x.append(bind[i][1])
            ave_y.append(nobind[i][0])
            var_y.append(nobind[i][1])

        # print(var_x, var_y)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)  # figsize指定总体画布的大小
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('ave', fontsize=20)
        ax1.set_xlabel('Bind', fontsize=20)
        ax1.set_ylabel('NoBind', fontsize=20)
        ax1.scatter(ave_x, ave_y, color='g')
        ax1.plot([0, 20], [0, 20], color='black')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('var', fontsize=20)
        ax2.set_xlabel('Bind', fontsize=20)
        ax2.set_ylabel('NoBind', fontsize=20)
        ax2.scatter(var_x, var_y, color='g')
        ax2.plot([0, 6], [0, 6], color='black')
        # plt.show()

    def covariance(self):
        bind = np.load(self.csv_path + './bind.npy', allow_pickle=True)
        bind = bind[500:]
        nobind = np.load(self.csv_path + './nobind.npy', allow_pickle=True)
        nobind = nobind[500:]

        bind_var = np.cov(m=bind, rowvar=False)
        nobind_var = np.cov(m=nobind, rowvar=False)

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)

        ax1 = fig.add_subplot(1, 2, 1)
        x_major_locator = MultipleLocator(20)  # 把x轴的刻度间隔设置为20，并存在变量里
        y_major_locator = MultipleLocator(20)  # 把y轴的刻度间隔设置为20，并存在变量里
        ax1.xaxis.set_major_locator(x_major_locator)
        ax1.yaxis.set_major_locator(y_major_locator)
        ax1.set_title('Bind', fontsize=20)
        ax1.set_xlabel('cp', fontsize=20)
        ax1.set_ylabel('cp', fontsize=20)
        im1 = ax1.imshow(bind_var, cmap="viridis")
        fig.colorbar(im1, pad=0.03)  # 设置颜色条

        ax2 = fig.add_subplot(1, 2, 2)
        x_major_locator = MultipleLocator(20)  # 把x轴的刻度间隔设置为20，并存在变量里
        y_major_locator = MultipleLocator(20)  # 把y轴的刻度间隔设置为20，并存在变量里
        ax2.xaxis.set_major_locator(x_major_locator)
        ax2.yaxis.set_major_locator(y_major_locator)
        ax2.set_title('NoBind', fontsize=20)
        ax2.set_xlabel('cp', fontsize=20)
        ax2.set_ylabel('cp', fontsize=20)
        im2 = ax2.imshow(nobind_var, cmap="viridis")
        fig.colorbar(im2, pad=0.03)  # 设置颜色条

        plt.show()
        # save
        # np.savetxt('./bind_var.csv', bind_var, delimiter=',', fmt="%.5f")
        # np.savetxt('./nobind_var.csv', nobind_var, delimiter=',', fmt="%.5f")

    def PCA_dis(self):
        bind = np.load('./Bind.npy', allow_pickle=True)
        bind = bind[500:]
        nobind = np.load('./Nobind.npy', allow_pickle=True)
        nobind = nobind[500:]

        print(bind.shape)
        # print(bind.shape)
        # print(nobind.shape)
        meanVal = np.mean(bind, axis=0)  # 按列求均值，即求各个特征的均值
        newData = bind - meanVal
        # print(meanVal.shape)
        covMat = np.cov(newData, rowvar=False)

        eigVals, eigVects = np.linalg.eig(np.mat(covMat))

        print(eigVals)

        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序,返回值为索引
        # print(eigValIndice)

        n_eigValIndice = eigValIndice[-1:-(30 + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
        lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
        # print(lowDDataMat.shape)
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
        # print(reconMat)
        # print(covMat)

    def PCA_dih(self):
        result_A_psi_bind = np.squeeze(np.load(self.csv_path + 'result_A_psi_bind.npy', allow_pickle=True))
        result_B_psi_bind = np.squeeze(np.load(self.csv_path + 'result_B_psi_bind.npy', allow_pickle=True))
        result_A_phi_bind = np.squeeze(np.load(self.csv_path + 'result_A_phi_bind.npy', allow_pickle=True))
        result_B_phi_bind = np.squeeze(np.load(self.csv_path + 'result_B_phi_bind.npy', allow_pickle=True))

        result_A_psi_nobind = np.squeeze(np.load(self.csv_path + 'result_A_psi_nobind.npy', allow_pickle=True))
        result_B_psi_nobind = np.squeeze(np.load(self.csv_path + 'result_B_psi_nobind.npy', allow_pickle=True))
        result_A_phi_nobind = np.squeeze(np.load(self.csv_path + 'result_A_phi_nobind.npy', allow_pickle=True))
        result_B_phi_nobind = np.squeeze(np.load(self.csv_path + 'result_B_phi_nobind.npy', allow_pickle=True))
        # 数据组织形式：(result_A_phi_bind, result_A_psi_bind, result_B_phi_bind, result_B_psi_bind)
        nobind = np.hstack((result_A_phi_nobind, result_A_psi_nobind, result_B_phi_nobind, result_B_psi_nobind))

        meanVal = np.mean(nobind, axis=0)  # 按列求均值，即求各个特征的均值
        newData = nobind - meanVal

        covMat = np.cov(newData, rowvar=False)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))

        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序,返回值为索引
        n_eigValIndice = eigValIndice[-1:-(30 + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量

        first_vect = np.abs(n_eigVect[:,0])
        first_val = eigVals[n_eigValIndice][0]
        print(np.where(first_vect > 0.07)[0] % 267)

    def Distance(self):
        bind = np.load(self.csv_path + 'bind_dis.npy', allow_pickle=True)
        nobind = np.load(self.csv_path + 'nobind_dis.npy', allow_pickle=True)
        print(bind.shape)
        # test = np.load(self.csv_path + 'iptg.npy', allow_pickle=True)

        bind_mean = np.mean(bind, axis=0)  # (340,)
        nobind_mean = np.mean(nobind, axis=0)  # (340,)

        bind_cov = np.cov(bind - bind_mean, rowvar=False)
        nobind_cov = np.cov(nobind - nobind_mean,
                            rowvar=False)  # np.mat((nobind - nobind_mean).T) * np.mat(nobind - nobind_mean)/4499

        Sw = np.mat(bind_cov + nobind_cov)
        w = Sw.I * np.mat(bind_mean - nobind_mean).T  # .I为求逆
        np.save(self.csv_path + "w_dis.npy", w)
        bind_map = bind * w
        nobind_map = nobind * w
        # test_map = test * w

        w_abs = np.absolute(w)
        w_sort = np.sort(w_abs, axis=0)
        cv_index = np.where(w_abs > 10)
        contact_li = list(np.load(self.csv_path + "aa_contact.npy", allow_pickle=True).item())
        contact_li.sort()
        cv = [contact_li[i] for i in cv_index[0]]
        np.save(self.csv_path + "cv.npy", cv)

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Binary_categories', fontsize=20)
        ax1.set_xlabel('LDA_value_Dis', fontsize=20)
        ax1.set_ylabel("frequency", fontsize=20)
        ax1.hist(np.array(bind_map), bins=100)
        ax1.hist(np.array(nobind_map), bins=100)
        # ax1.scatter(np.array(test_map), np.ones(shape=(5000,)), marker='D')
        # plt.savefig(self.output + '{0}_{1}.png'.format('LDA', self.data_name))
        plt.show()

    def Dihedral(self):
        result_A_psi_bind = np.squeeze(np.load(self.csv_path + 'result_A_psi_bind.npy', allow_pickle=True))
        result_B_psi_bind = np.squeeze(np.load(self.csv_path + 'result_B_psi_bind.npy', allow_pickle=True))
        result_A_phi_bind = np.squeeze(np.load(self.csv_path + 'result_A_phi_bind.npy', allow_pickle=True))
        result_B_phi_bind = np.squeeze(np.load(self.csv_path + 'result_B_phi_bind.npy', allow_pickle=True))

        result_A_psi_nobind = np.squeeze(np.load(self.csv_path + 'result_A_psi_nobind.npy', allow_pickle=True))
        result_B_psi_nobind = np.squeeze(np.load(self.csv_path + 'result_B_psi_nobind.npy', allow_pickle=True))
        result_A_phi_nobind = np.squeeze(np.load(self.csv_path + 'result_A_phi_nobind.npy', allow_pickle=True))
        result_B_phi_nobind = np.squeeze(np.load(self.csv_path + 'result_B_phi_nobind.npy', allow_pickle=True))
        # 数据组织形式：A_Phi, A_Psi, B_Phi, B_Psi
        bind = np.hstack((result_A_phi_bind, result_A_psi_bind, result_B_phi_bind, result_B_psi_bind))
        nobind = np.hstack((result_A_phi_nobind, result_A_psi_nobind, result_B_phi_nobind, result_B_psi_nobind))

        bind = bind[500:]
        nobind = nobind[500:]

        bind_mean = np.mean(bind, axis=0)  # (302,)
        nobind_mean = np.mean(nobind, axis=0)  # (302,)

        diff = bind - bind_mean

        bind_cov = np.cov(bind - bind_mean, rowvar=False)
        nobind_cov = np.cov(nobind - nobind_mean, rowvar=False)

        Sw = np.mat(bind_cov + nobind_cov)  # 要求对应特征一一对应
        w = Sw.I * np.mat(bind_mean - nobind_mean).T  # .I为求逆
        np.save(self.csv_path + "w.npy", w)
        bind_map = bind * w
        nobind_map = nobind * w

        # ax1.scatter(np.array(test_map), np.ones(shape=(5000,)), marker='D')
        weightIndice = np.argsort(np.abs(w), axis=None)  # 对权重从小到大排序,返回值为索引
        n_weightIndice = weightIndice[0,-1:-(50 + 1):-1]  # 最大的n个特征值的下标, 30个
        n_weightIndice_indice = np.argsort(n_weightIndice)

        n_weightIndice_order = n_weightIndice[0,n_weightIndice_indice] # 前n个最大的权重索引，顺序为特征顺序

        new_w = np.squeeze(w[:,0][n_weightIndice_order])

        new_bind = np.squeeze(bind[:, n_weightIndice_order])
        new_nobind = np.squeeze(nobind[:, n_weightIndice_order])

        new_bind_map = new_bind * new_w.T
        new_nobind_map = new_nobind * new_w.T

        np.save("./dihedral_new_w.npy", new_w.T)
        np.save("./dihedral_n_weightIndice_order.npy", n_weightIndice_order)

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Binary_categories', fontsize=20)
        ax1.set_xlabel('scatter', fontsize=20)
        ax1.set_ylabel("no meaning", fontsize=20)
        ax1.scatter(np.array(new_bind_map), np.ones(shape=(4501,)), marker='x')
        ax1.scatter(np.array(new_nobind_map), np.ones(shape=(4501,)), marker='+')

        # print(w[:,0][n_weightIndice])
        plt.show()
        # plt.savefig(self.output + '{0}.png'.format('Dihedral'))

    def Dihedral_sc(self):
        # 数据组织形式：A_Phi, A_Psi, B_Phi, B_Psi
        bind = np.load("./bind_sc.npy", allow_pickle=True)
        nobind = np.load("./nobind_sc.npy", allow_pickle=True)
        print(bind.shape)

        bind_mean = np.mean(bind, axis=0)  # (340,)
        nobind_mean = np.mean(nobind, axis=0)  # (340,)

        bind_cov = np.cov(bind - bind_mean, rowvar=False)
        nobind_cov = np.cov(nobind - nobind_mean, rowvar=False)

        Sw = np.mat(bind_cov + nobind_cov)  # 要求对应特征一一对应
        w = Sw.I * np.mat(bind_mean - nobind_mean).T  # .I为求逆
        # print(w.shape) (2136, 1)
        weightIndice = np.argsort(np.abs(w), axis=None)  # 对权重从小到大排序,返回值为索引
        # print(weightIndice.shape) (1, 2136)
        n_weightIndice = weightIndice[0, -1:-(300 + 1):-1]  # 最大的n个特征值的下标
        n_weightIndice_indice = np.argsort(n_weightIndice)

        n_weightIndice_order = n_weightIndice[0, n_weightIndice_indice]  # 前n个最大的权重索引，顺序为特征顺序

        new_w = np.squeeze(w[:, 0][n_weightIndice_order])

        new_bind = np.squeeze(bind[:, n_weightIndice_order])
        new_nobind = np.squeeze(nobind[:, n_weightIndice_order])

        new_bind_map = new_bind * new_w.T
        new_nobind_map = new_nobind * new_w.T

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Binary_categories', fontsize=20)
        ax1.set_xlabel('LDA_value_Dih', fontsize=20)
        ax1.set_ylabel("frequency", fontsize=20)
        ax1.hist(np.array(new_bind_map), bins=100)
        ax1.hist(np.array(new_nobind_map), bins=100)

        plt.show()
        # np.save(self.csv_path + "w_dih.npy", w)
        # np.save("./dihedral_new_w.npy", new_w.T)
        # np.save("./dihedral_n_weightIndice_order.npy", n_weightIndice_order)

    def single_LDA_Dis(self, pdb_path, uni):
        a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = self.readHeavyAtom(pdb_path)
        data_df = self.calContact(a_atom_cor, b_atom_cor, a_atom_nam=a_atom_nam, b_atom_nam=b_atom_nam, filename=uni,
                        save_dis=False)[1]
        aa_contact = list(np.load('./aa_contact.npy', allow_pickle=True).item())
        aa_contact.sort()
        contact_dif = np.zeros(shape=(1, len(aa_contact)))

        # dataframe = pd.read_csv('./{0}.csv'.format(uni))
        new_col = []
        new_index = []
        # 去除原子序数
        for col in data_df.columns:
            record = col.split('-')
            new_col.append(record[0] + '-' + record[1] + '-' + record[2])
        for row in data_df.index:
            record = row.split('-')
            new_index.append(record[0] + '-' + record[1] + '-' + record[2])
        data_df.columns = new_col
        data_df.index = new_index

        # print(data_df)
        # 构建含有CB的cp
        for l in range(len(aa_contact)):
            cp = aa_contact[l]
            a_rec = cp[0].split('-')
            b_rec = cp[1].split('-')
            a_atom = a_rec[0] + '-' + a_rec[1] + '-' + ['CB' if a_rec[1] != 'GLY' else 'CA'][0]
            b_atom = b_rec[0] + '-' + b_rec[1] + '-' + ['CB' if b_rec[1] != 'GLY' else 'CA'][0]
            contact_dif[0][l] = data_df[a_atom][b_atom]

        # np.save("./NPY_dis/{0}.npy".format(frame), contact_dif)
        w = np.load('./w_dis.npy', allow_pickle=True)
        print(contact_dif)
        print(contact_dif.shape)
        value = (np.mat(contact_dif) * w).tolist()[0][0]
        print(value)
        return value

    def single_LDA_dih(self, pdb_path, frame):
        # 行：每一帧
        # 列：每一个二面角
        result_A_phi = []
        result_A_psi = []
        result_B_phi = []
        result_B_psi = []

        phi = self.gather_dihedral_atom(pdb_path, type="Phi")
        result_A_phi.append(phi[0][1:])  # 去除掉360，防止逆矩阵为奇异阵
        result_B_phi.append(phi[1][1:])

        psi = self.gather_dihedral_atom(pdb_path, type="Psi")
        result_A_psi.append(psi[0][:-1])  # 去除掉360，防止逆矩阵为奇异阵
        result_B_psi.append(psi[1][:-1])

        result_A_phi = np.array(result_A_phi)
        result_A_psi = np.array(result_A_psi)
        result_B_phi = np.array(result_B_phi)
        result_B_psi = np.array(result_B_psi)

        result = np.hstack((result_A_phi, result_A_psi, result_B_phi, result_B_psi))

        result = np.hstack((np.sin((result * np.pi) / 180), np.cos((result * np.pi) / 180)))
        np.save("./NPY_dih/{0}.npy".format(frame), result)


        new_w = np.load('./dihedral_new_w.npy', allow_pickle=True)
        n_weightIndice_order = np.load('./dihedral_n_weightIndice_order.npy', allow_pickle=True)
        new_sample = np.squeeze(result[:, n_weightIndice_order], axis=1)

        value = (np.mat(new_sample) * new_w).tolist()[0][0]
        # print(value)
        return value


    def LDA_trend(self):
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('LDA_trend', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel("LDA", fontsize=20)
        ax1.plot()  # ###############!!!!!!!
        plt.show()

    def rmsd_plot_gmx(self):  # 单位为Å
        path = "/Users/erik/Desktop/Pareto/reverse/5th/"
        filename = "rmsd_weight.xvg"
        frame = 0
        rms = []
        with open(path+filename) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    rms.append(float(li[1])*10)  # Å
                    frame += 1

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel("RMSD(Å)", fontsize=20)
        ax1.scatter(range(frame), rms, s=.3)
        plt.show()
        np.save(path+"rmsd.npy", np.array(rms))

    def rmsd_plot_vmd(self):  # 单位为Å
        path = "/Users/erik/Desktop/Pareto/reverse/4th/rmsd.xvg"
        frame = []
        rms = []
        with open(path, 'r') as file:
            for i in file.readlines():
                record = i.split()
                if len(record) != 0 and record[0] not in ['frame', '0']:
                    print(record)
                    frame.append(float(record[0]))
                    rms.append(float(record[1]))

        np.save(self.vmd_rmsd_path + "rms.npy", np.array(rms))
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel("RMSD(Å)", fontsize=20)
        ax1.plot(frame, rms)
        plt.show()

    def rmsf_plot(self):  # 单位为Å
        file_name = 'rmsf_res'
        target_file = self.vmd_rmsd_path + file_name + ".xvg"
        x = [[], []]
        y = [[], []]
        chain_id = -1
        with open(target_file) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    res_id = float(li[0])
                    if res_id == 1:
                        chain_id += 1
                    x[chain_id].append(res_id)
                    y[chain_id].append(float(li[1])*10)

        # np.save(self.vmd_rmsd_path + file_name + ".npy", np.array(y))
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('residue', fontsize=20)
        ax1.set_ylabel("RMSF(Å)", fontsize=20)
        ax1.plot(x[0], y[0])  # 4.1G
        ax1.plot(x[1], y[1])  # NuMA

        # hydrophobic_index
        ax1.scatter(self.hydrophobic_index[0], [y[0][i - 1] for i in self.hydrophobic_index[0]], color="black")
        ax1.scatter(self.hydrophobic_index[1], [y[1][i - 1] for i in self.hydrophobic_index[1]], color="black")

        # hydrophilic_index
        ax1.scatter(self.hydrophilic_index[0], y[0][self.hydrophilic_index[0] - 1], color="red")

        # either
        ax1.scatter(self.either_index, [y[0][i - 1] for i in self.either_index], color="green")

        plt.show()

    def statistics(self):
        indicator = ['potential_kJ_mol', 'temperature', 'pressure', 'density', 'RMSD-CA', 'RMSF(nm)']
        #                   0                     1           2         3         4          5
        for i in range(3):
            target_file = self.vmd_rmsd_path + '{0}.xvg'.format(indicator[i])
            x = []
            y = []
            with open(target_file) as f:
                for j in f.readlines():
                    record = j.strip()
                    if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                        break
                    if record[:12] == "@ s0 legend ":
                        title = record.split()[-1]
                    if record[0] not in ["#", "@"]:
                        li = record.split()
                        x.append(float(li[0]))
                        y.append(float(li[1]))

            fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_title(title, fontsize=20)
            ax1.set_xlabel('frame', fontsize=20)
            ax1.set_ylabel(indicator[i], fontsize=20)
            ax1.plot(range(len(x)), y)
            plt.savefig(self.output + '{0}_{1}.png'.format(indicator[i], self.data_name))
            plt.cla()
        # plt.show()

    def aa_contact(self):
        # 读取
        path = self.output + "total_contact_{0}.npy".format(self.data_name)
        cp = list(np.load(path, allow_pickle=True).item())
        # 排序
        cp.sort()
        aa_cp = set()
        for i in cp:
            record1 = i[0].split('-')
            record2 = i[1].split('-')
            # formatting
            aa_cp.add((record1[0] + '-' + record1[1], record2[0] + '-' + record2[1]))
        np.save(self.output + "aa_contact_{0}.npy".format(self.data_name), aa_cp)

    def extractFeat(self):
        train_data = []
        # contact_li都是原始状态的残基名称，不再有ASH、HID等质子化名称 whole_cp
        #                                        total_contact
        contact_li = list(np.load(self.output + "aa_contact.npy", allow_pickle=True).item())
        # 保证contact_pair顺序一致，排序
        contact_li.sort()
        # 遍历
        for i in range(self.startFrame, self.endFrame):
            csv_path = self.csv_path + self.csv_name.format(i * self.Interval)
            print("Reading", csv_path)
            dis_df = pd.read_csv(csv_path)

            # 删除原子序号信息，便于随机访问  62-LEU-N-1
            new_col = ['Unnamed: 0']
            new_index = []
            for col in dis_df.columns[1:]:
                record = col.split('-')
                new_col.append(record[0] + '-' + record[1] + '-' + record[2])

            for index in dis_df[dis_df.columns[0]]:
                record = index.split('-')
                new_index.append(record[0] + '-' + record[1] + '-' + record[2])

            dis_df.columns = new_col
            dis_df.index = new_index

            feat = []
            for cp in contact_li:
                print(cp)
                # 因为gly没有CA，所以换为CA
                a_index = cp[0] + '-' + ['CB' if cp[0][-3:] != 'GLY' else 'CA'][0]
                b_index = cp[1] + '-' + ['CB' if cp[1][-3:] != 'GLY' else 'CA'][0]
                feat.append(dis_df[a_index][b_index])  # 先列后行
            train_data.append(feat)  # 文件csv和数组索引差1

        train_array_x = np.array(train_data)
        print(train_array_x)
        np.save(self.output + "{0}.npy".format(self.data_name), train_array_x)

    def merge_dis(self):
        nobind_1 = pd.read_csv("./work1/contact_dif_CB_NoIptg_ProNoBindDna_work1.csv")
        bind_1 = pd.read_csv("./work1/contact_dif_CB_NoIptg_ProBindDna_work1.csv")

        nobind_2 = pd.read_csv("./work2/contact_dif_CB_NoIptg_Nobind.csv")
        bind_2 = pd.read_csv("./work2/contact_dif_CB_NoIptg_Bind.csv")

        nobind_3 = pd.read_csv("./work3/contact_dif_CB_NoIptg_NoBind.csv")
        bind_3 = pd.read_csv("./work3/contact_dif_CB_NoIptg_Bind.csv")

        nobind_1 = nobind_1.values[:,1:]
        bind_1 = bind_1.values[:, 1:]

        nobind_2 = nobind_2.values[:, 1:]
        bind_2 = bind_2.values[:, 1:]

        nobind_3 = nobind_3.values[:, 1:]
        bind_3 = bind_3.values[:, 1:]

        Bind = np.vstack((bind_1, bind_2, bind_3))
        noBind = np.vstack((nobind_1, nobind_2, nobind_3))

        np.save("./bind.npy", Bind)
        np.save("./nobind.npy", noBind)

    def merge_dih(self):
        bind_A_phi_1 = np.load("./work1/bind/result_A_phi.npy", allow_pickle=True)
        bind_A_psi_1 = np.load("./work1/bind/result_A_psi.npy", allow_pickle=True)
        bind_B_phi_1 = np.load("./work1/bind/result_B_phi.npy", allow_pickle=True)
        bind_B_psi_1 = np.load("./work1/bind/result_B_psi.npy", allow_pickle=True)
        bind_1 = np.hstack((bind_A_phi_1, bind_A_psi_1, bind_B_phi_1, bind_B_psi_1))

        bind_A_phi_2 = np.load("./work2/bind/result_A_phi.npy", allow_pickle=True)
        bind_A_psi_2 = np.load("./work2/bind/result_A_psi.npy", allow_pickle=True)
        bind_B_phi_2 = np.load("./work2/bind/result_B_phi.npy", allow_pickle=True)
        bind_B_psi_2 = np.load("./work2/bind/result_B_psi.npy", allow_pickle=True)
        bind_2 = np.hstack((bind_A_phi_2, bind_A_psi_2, bind_B_phi_2, bind_B_psi_2))

        bind_A_phi_3 = np.load("./work3/bind/result_A_phi.npy", allow_pickle=True)
        bind_A_psi_3 = np.load("./work3/bind/result_A_psi.npy", allow_pickle=True)
        bind_B_phi_3 = np.load("./work3/bind/result_B_phi.npy", allow_pickle=True)
        bind_B_psi_3 = np.load("./work3/bind/result_B_psi.npy", allow_pickle=True)
        bind_3 = np.hstack((bind_A_phi_3, bind_A_psi_3, bind_B_phi_3, bind_B_psi_3))

        nobind_A_phi_1 = np.load("./work1/nobind/result_A_phi.npy", allow_pickle=True)
        nobind_A_psi_1 = np.load("./work1/nobind/result_A_psi.npy", allow_pickle=True)
        nobind_B_phi_1 = np.load("./work1/nobind/result_B_phi.npy", allow_pickle=True)
        nobind_B_psi_1 = np.load("./work1/nobind/result_B_psi.npy", allow_pickle=True)
        nobind_1 = np.hstack((nobind_A_phi_1, nobind_A_psi_1, nobind_B_phi_1, nobind_B_psi_1))

        nobind_A_phi_2 = np.load("./work2/nobind/result_A_phi.npy", allow_pickle=True)
        nobind_A_psi_2 = np.load("./work2/nobind/result_A_psi.npy", allow_pickle=True)
        nobind_B_phi_2 = np.load("./work2/nobind/result_B_phi.npy", allow_pickle=True)
        nobind_B_psi_2 = np.load("./work2/nobind/result_B_psi.npy", allow_pickle=True)
        nobind_2 = np.hstack((nobind_A_phi_2, nobind_A_psi_2, nobind_B_phi_2, nobind_B_psi_2))

        nobind_A_phi_3 = np.load("./work3/nobind/result_A_phi.npy", allow_pickle=True)
        nobind_A_psi_3 = np.load("./work3/nobind/result_A_psi.npy", allow_pickle=True)
        nobind_B_phi_3 = np.load("./work3/nobind/result_B_phi.npy", allow_pickle=True)
        nobind_B_psi_3 = np.load("./work3/nobind/result_B_psi.npy", allow_pickle=True)
        nobind_3 = np.hstack((nobind_A_phi_3, nobind_A_psi_3, nobind_B_phi_3, nobind_B_psi_3))

        bind = np.squeeze(np.vstack((bind_1, bind_2, bind_3)))
        nobind = np.squeeze(np.vstack((nobind_1, nobind_2, nobind_3)))

        bind = np.hstack((np.sin((bind * np.pi) / 180), np.cos((bind * np.pi) / 180)))
        nobind = np.hstack((np.sin((nobind * np.pi) / 180), np.cos((nobind * np.pi) / 180)))

        np.save("./bind_sc.npy", bind)
        np.save("./nobind_sc.npy", nobind)

    def sasa_sf(self, path):  # 计算单一溶液可及性面积
        result = []
        score = 110
        with open(path, 'r') as f:
            for i in f.readlines():
                record = i.strip()
                if record[0:3] == 'ASG':
                    aa_name = record[5:8]
                    result.append(self.relative_SASA(aa_name, float(record[64:69])))

        hydrophobic_index = [15, 19, 33, 37, 49, 52, 81, 82, 84, 86]
        hydrophobic_threshold = 0.36

        either_index = [36, 40, 44]

        hydrophilic_index = [35]
        hydrophilic_threshold = 0.36

        for k in hydrophobic_index:
            if result[k] > hydrophobic_threshold:
                print(k)
                score -= 10
        for j in hydrophilic_index:
            if result[j] <= hydrophilic_threshold:
                print(j)
                score -= 10

        # return result
        return score

    def sasa(self):
        """path_dir = "/Users/erik/Desktop/MD_WWN/REMD/SASA/"
        num = 16
        series = dict()

        for i in range(1,num+1):
            for j in range(1, 2001):
                path = path_dir + "{0}/sasa_md.pdb.{1}".format(i, j)
                if str(i) not in series.keys():
                    series[str(i)] = [self.sasa_sf(path)]
                else:
                    series[str(i)].append(self.sasa_sf(path))
        np.save("./REMD_16_SASA.npy", series)"""

        num = 16
        series = np.load("./REMD_16_SASA.npy", allow_pickle=True).item()

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, num + 1):
            ax1 = fig.add_subplot(4, 4, k)  # 左上角第一个开始填充，从左到右
            ax1.set_title('score of REMD', fontsize=2)
            ax1.set_xlabel('frames(5ps/f)', fontsize=2)
            ax1.set_ylabel("score", fontsize=2)
            ax1.scatter(range(2000), series[str(k)], s=.1)

        plt.show()

        return np.array(series)


    def relative_SASA(self, aa_name, SASA):
        return SASA/self.sasa_max[self.processIon(aa_name)]

    def sasa_cluster(self):
        result = []
        num = 15
        for i in range(1, num+1):
            result.append(self.sasa_sf("/Users/erik/Desktop/MD_WWN/SASA/sasa_cluster_{0}.pdb".format(i)))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('clusters in paper', fontsize=20)
        ax1.set_xlabel('cluster_serial', fontsize=20)
        ax1.set_ylabel("score", fontsize=20)
        ax1.scatter([i for i in range(1, num+1)], result)  # ###############!!!!!!!

        # plt.savefig('sasa.png')
        plt.show()

    def dihdis_trend(self): # reverse 3; forward 1  4.24

        target = "dih"
        direction = "forward"
        i_start = 1
        i_end = 3
        data = []

        if target == "dis":
            for i in range(i_start, i_end + 1):
                path = "/Users/erik/Desktop/Pareto/{1}/{0}th/".format(i, direction)
                for n in range(1, 11):
                    dis_value = np.load(path + "dis_{0}.npy".format(n), allow_pickle=True)
                    data += dis_value.tolist()

        if target == "dih":
            for i in range(i_start, i_end + 1):
                path = "/Users/erik/Desktop/Pareto/{1}/{0}th/".format(i, direction)
                for n in range(1, 11):
                    dih_value = np.load(path + "dih_{0}.npy".format(n), allow_pickle=True)
                    data += dih_value.tolist()

        print(len(data))
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('{0}_trend'.format(target), fontsize=20)
        ax1.set_xlabel('frames', fontsize=20)
        ax1.set_ylabel(target, fontsize=20)
        ax1.scatter(range(len(data)), data, s=.3)

        plt.show()

    def relative_sasa_statistics(self, aa):
        result = []
        for k in range(1, 5000):
            path = "/Users/erik/Desktop/MD_WWN/test_100ns/SASA/sasa_md.pdb." + str(k)
            with open(path, 'r') as f:
                for i in f.readlines():
                    record = i.strip()
                    if record[0:3] == 'ASG' and record[5:8] == aa:
                        aa_name = record[5:8]
                        result.append(self.relative_SASA(aa_name, float(record[64:69])))

        plt.hist(result, bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.show()

    def rmsd_between(self, name1, name2, prefix_path):
        pm.finish_launching()
        pm.cmd.load(prefix_path + '/' + name1 + ".pdb")
        pm.cmd.load(prefix_path + '/' + name2 + ".pdb")

        pm.cmd.select("bb1", "(name CA+N+C) & " + name1)
        pm.cmd.select("bb2", "(name CA+N+C) & " + name2)

        return pm.cmd.rms("bb1","bb2")


    def rmsd_batch(self):
        result = []
        prefix_path = "./frame"
        for i in range(5001):
            rmsd_value = self.rmsd_between("md0", "md"+str(i), prefix_path)
            print(rmsd_value)
            result.append(rmsd_value)

        np.save("./rms.npy", np.array(result))

    def add_chainID(self, file_path):
        chain_ID = ["A", "B"]
        n = -1
        current_aa = ""
        with open(file_path, 'r') as f:
            for i in f.readlines():
                record = i.strip()
                atom = record[:4].strip()
                if atom == "TEM":
                    break
                if atom != "ATOM":  # 检测ATOM起始行
                    continue

                resName = self.processIon(record[17:20].strip())  # PRO, 已处理过质子化条件
                resSeq = int(record[22:26].strip())
                if resSeq == 62 and current_aa != resName:     # 1或62，取决于是我的体系还是小红师姐的体系
                    n += 1
                record = record[:21] + chain_ID[n] + record[22:]
                current_aa = resName
                print(record)

    def read_extract_cor(self):
        atom_cor = []
        with open("./extract_cor.txt", 'r') as f:
            for i in f.readlines():
                record = [float(j) for j in i.strip().split()]
                atom_cor.append(record)

        atom_cor = np.array(atom_cor)
        print(atom_cor)
        print(atom_cor.shape)

    def Pareto_surface(self, average=False):
        path = "/Users/erik/Desktop/Pareto/reverse/6th/"
        dis = []
        dih = []

        for i in range(1, 11):
            dis_value = np.load(path + "dis_{0}.npy".format(i), allow_pickle=True)
            dih_value = np.load(path + "dih_{0}.npy".format(i), allow_pickle=True)

            dis += dis_value.tolist()
            dih += dih_value.tolist()

        # print("0 frame dis: ", dis[0])
        # print("0 frame dih: ", dih[0])

        if average:
            a = np.exp(-20 / 60)  # dt/τ
            EMA_dis = []
            EMA_dih = []
            cur_dis = dis[0]
            cur_dih = dih[0]
            EMA_dis.append(cur_dis)
            EMA_dih.append(cur_dih)
            for i in range(1, len(dis)):
                cur_dis = cur_dis * (1-a) + dis[i] * a
                cur_dih = cur_dih * (1-a) + dih[i] * a
                EMA_dis.append(cur_dis)
                EMA_dih.append(cur_dih)

            dis = EMA_dis
            dih = EMA_dih

        # print(dis)

        """stage1_dis = np.load(path + "dis_{0}.npy".format(2), allow_pickle=True)
        stage1_dih = np.load(path + "dih_{0}.npy".format(2), allow_pickle=True)

        stage2_dis = np.load(path + "dis_{0}.npy".format(4), allow_pickle=True)
        stage2_dih = np.load(path + "dih_{0}.npy".format(4), allow_pickle=True)

        stage3_dis = np.load(path + "dis_{0}.npy".format(6), allow_pickle=True)
        stage3_dih = np.load(path + "dih_{0}.npy".format(6), allow_pickle=True)

        stage4_dis = np.load(path + "dis_{0}.npy".format(8), allow_pickle=True)
        stage4_dih = np.load(path + "dih_{0}.npy".format(8), allow_pickle=True)

        stage5_dis = np.load(path + "dis_{0}.npy".format(10), allow_pickle=True)
        stage5_dih = np.load(path + "dih_{0}.npy".format(10), allow_pickle=True)"""


        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Pareto_surface', fontsize=20)
        ax1.set_xlabel('dis', fontsize=20)
        ax1.set_ylabel("dih", fontsize=20)
        ax1.scatter(dis, dih)
        """ax1.scatter(stage1_dis, stage1_dih, color="yellow")
        ax1.scatter(stage2_dis, stage2_dih, color="pink")
        ax1.scatter(stage3_dis, stage3_dih, color="red")
        ax1.scatter(stage4_dis, stage4_dih, color="black")
        ax1.scatter(stage5_dis, stage5_dih, color="green")"""
        plt.show()
        # self.pick_frame(np.array(dis), np.array(dih))

    def resemble_average(self, dis, dih):
        a = np.exp(-20 / 60)  # dt/τ

        EMA_dis = []
        EMA_dih = []
        cur_dis = dis[0]
        cur_dih = dih[0]
        EMA_dis.append(cur_dis)
        EMA_dih.append(cur_dih)
        for i in range(1, len(dis)):
            cur_dis = cur_dis * (1 - a) + dis[i] * a
            cur_dih = cur_dih * (1 - a) + dih[i] * a
            EMA_dis.append(cur_dis)
            EMA_dih.append(cur_dih)

        return dis, dih

    def read_rmsd_gmx(self, path):
        # path = "/Users/erik/Desktop/Pareto/reverse/4th/"
        # filename = "rmsd.xvg"
        frame = 0
        rms = []
        with open(path) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    rms.append(float(li[1]) * 10)  # Å
                    frame += 1

        return rms

    def Pathway(self):
        interval = 2500  # 每50ns更换颜色


        for_i_start = 1
        for_i_end = 1
        for_dis = []
        for_dih = []
        for_rms = []

        rev_j_start = 3
        rev_j_end = 3
        rev_dis = []
        rev_dih = []
        rev_rms = []
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title('', fontsize=20)
        ax1.set_xlabel('dis', fontsize=20)
        ax1.set_ylabel("dih", fontsize=20)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title('RMSD_forward', fontsize=20)
        ax2.set_xlabel('frames', fontsize=20)
        ax2.set_ylabel("rmsd(Å)", fontsize=20)

        ax3 = fig.add_subplot(2, 1, 2)
        ax3.set_title('RMSD_reverse', fontsize=20)
        ax3.set_xlabel('frames', fontsize=20)
        ax3.set_ylabel("rmsd(Å)", fontsize=20)

        for i in range(for_i_start, for_i_end + 1):
            path = "/Users/erik/Desktop/Pareto/forward/{0}th/".format(i)
            for_rms += self.read_rmsd_gmx(path + "rmsd.xvg")
            for n in range(1, 11):
                dis_value = np.load(path + "dis_{0}.npy".format(n), allow_pickle=True)
                dih_value = np.load(path + "dih_{0}.npy".format(n), allow_pickle=True)

                for_dis += dis_value.tolist()
                for_dih += dih_value.tolist()

        for j in range(rev_j_start, rev_j_end + 1):
            path = "/Users/erik/Desktop/Pareto/reverse/{0}th/".format(j)
            rev_rms += self.read_rmsd_gmx(path + "rmsd.xvg")
            for n in range(1, 11):
                dis_value = np.load(path + "dis_{0}.npy".format(n), allow_pickle=True)
                dih_value = np.load(path + "dih_{0}.npy".format(n), allow_pickle=True)

                rev_dis += dis_value.tolist()
                rev_dih += dih_value.tolist()

        for_dis, for_dih = self.resemble_average(for_dis, for_dih)
        rev_dis, rev_dih = self.resemble_average(rev_dis, rev_dih)

        for k in range(int(len(rev_dis)/interval)):
            ax1.scatter(rev_dis[interval*k:interval*(k+1)], rev_dih[interval*k:interval*(k+1)], s=.5)
            # ax3.scatter(range(interval * k, interval * (k + 1)), rev_rms[interval * k:interval * (k + 1)], s=.2)

        for k in range(int(len(for_dis)/interval)):
            ax1.scatter(for_dis[interval*k:interval*(k+1)], for_dih[interval*k:interval*(k+1)], s=.5)
            # ax2.plot(range(interval * k, interval * (k + 1)), for_rms[interval * k:interval * (k + 1)])

        """for k in range(int(len(for_rms) / interval)):
            ax2.plot(range(interval * k, interval * (k + 1)), for_rms[interval * k:interval * (k + 1)])

        for k in range(int(len(rev_rms) / interval)):
            ax3.plot(range(interval * k, interval * (k + 1)), rev_rms[interval * k:interval * (k + 1)])"""

        plt.show()
        # self.pick_frame(np.array(dis), np.array(dih))

    def pick_frame(self, dis_array, dih_array):
        index = np.where(-74 < dis_array)  # 元组，需要用索引取得
        for i in index[0]:
            if dih_array[i] > -740:
                print(i)
                print(dih_array[i])
                print(dis_array.shape)

    def process_pbc_cubic(self, origin_cor, atom_cor, pbc=None):
        DIM = 3
        hbox = []
        for i in range(DIM):
            hbox.append(0.5 * pbc[i][i])
        for n in range(atom_cor.shape[0]):
            for m in range(DIM-1, -1, -1):
                while atom_cor[n][m] - origin_cor[n][m] <= -hbox[m]:
                    print(n, m, atom_cor[n][m], origin_cor[n][m], atom_cor[n][m] - origin_cor[n][m])
                    for d in range(m+1):
                        atom_cor[n][d] += pbc[m][d]

                while atom_cor[n][m] - origin_cor[n][m] > hbox[m]:
                    print(n, m, atom_cor[n][m], origin_cor[n][m], atom_cor[n][m] - origin_cor[n][m])
                    for d in range(m+1):
                        atom_cor[n][d] -= pbc[m][d]
        return atom_cor

    def rmsf_plot_amber(self):
        path = self.vmd_rmsd_path + "rmsf_stage_2.data"
        Res_41G = []
        Res_numa = []
        AtomicFlx_41G = []
        AtomicFlx_numa = []

        Res = [Res_41G, Res_numa]
        AtomicFlx = [AtomicFlx_41G, AtomicFlx_numa]

        chainID = 0
        with open(path, 'r') as f:
            for i in f.readlines():
                record = i.strip()
                if record[0] != '#':
                    record = record.split()
                    resid = int(float(record[0]))
                    if resid == 68:
                        chainID += 1
                    Res[chainID].append(resid)
                    AtomicFlx[chainID].append(float(record[1]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('RMSF', fontsize=20)
        ax1.set_xlabel('Resid', fontsize=20)
        ax1.set_ylabel("rmsf(Å)", fontsize=20)
        ax1.plot(Res[0], AtomicFlx[0])  # ###############!!!!!!!
        ax1.plot(Res[1], AtomicFlx[1])

        # hydrophobic_index
        ax1.scatter(self.hydrophobic_index[0], [AtomicFlx[0][i - 1] for i in self.hydrophobic_index[0]], color="black")
        ax1.scatter(self.hydrophobic_index[1], [AtomicFlx[1][i - 1 - 67] for i in self.hydrophobic_index[1]], color="black")

        # hydrophilic_index
        ax1.scatter(self.hydrophilic_index[0], AtomicFlx[0][self.hydrophilic_index[0] - 1], color="red")

        # either
        ax1.scatter(self.either_index, [AtomicFlx[0][i - 1] for i in self.either_index], color="green")
        # plt.savefig('sasa.png')

        plt.show()

    def REMD_temperature_generation(self, start, end, replicas):
        if os.path.exists("./temperatures.dat"):
            os.system("rm temperatures.dat")
        T = []

        k = np.log(np.power(end/start, 1/7))
        print(k)

        with open("temperatures.dat", 'a') as f:
            for i in range(replicas):
                T.append(start*np.exp(k*i))
                f.write("%.1f" % float(start*np.exp(k*i)))  # 保留一位小数, 并且保证温度指数间隔
                f.write("\n")

        f.close()
        return T

    def sasa_statistics(self):
        """data = self.sasa()
        np.save("./sasa.npy", data)"""
        data = np.load("./sasa.npy", allow_pickle=True)
        # print(data.shape) # 50000,93
        data_mean = np.mean(data, axis=0)
        # print(data_mean.shape)  # 93,
        data_var = np.std(data, axis=0)
        # print(data_var.shape)

        # print(data_mean[15])
        # print(data_var[15])

        hydropho_mean = [data_mean[i - 1] for i in self.hydrophobic_index]
        hydropho_var = [data_var[i - 1] for i in self.hydrophobic_index]

        hydrophi_mean = [data_mean[i - 1] for i in self.hydrophilic_index]
        hydrophi_var = [data_var[i - 1] for i in self.hydrophilic_index]

        either_mean = [data_mean[i - 1] for i in self.either_index]
        either_var = [data_var[i - 1] for i in self.either_index]

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('SASA', fontsize=20)
        ax1.set_xlabel('Resid', fontsize=20)
        ax1.set_ylabel("relative_SASA", fontsize=20)

        ax1.errorbar([i+1 for i in range(data.shape[1])], data_mean, yerr=data_var, fmt="o")

        ax1.errorbar(self.hydrophobic_index, hydropho_mean, yerr=hydropho_var, fmt="o", color="black")
        ax1.errorbar(self.hydrophilic_index, hydrophi_mean, yerr=hydrophi_var, fmt="o", color="red")
        ax1.errorbar(self.either_index, either_mean, yerr=either_var, fmt="o", color="green")

        plt.show()

    def aaNumSASA(self):
        data = np.load("./sasa.npy", allow_pickle=True)
        data_mean = np.mean(data, axis=0)
        # print(data_mean.shape)

        a = 0  # <=0.36
        b = 0  # between 0.36 and 0.6
        c = 0  # =>0.6

        for i in data_mean:
            if i <= 0.36:
                a += 1
            elif i >= 0.6:
                c += 1
            else:
                b += 1

        print(a,b,c)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('SASA', fontsize=20)
        ax1.set_xlabel('Resid', fontsize=20)
        ax1.set_ylabel("relative_SASA", fontsize=20)
        ax1.bar(["<=0.36", "0.36<sasa<0.6", ">=0.6"], [a, b, c])

        plt.show()

    """def time_series_T(self):
        path = "/Users/erik/Downloads/remd_output/"
        num = 8
        temperatures = self.REMD_temperature_generation(269.5, 570.9, num)
        series = dict()
        current_replica = 1
        with open(path+"rem.log") as f:
            for j in f.readlines():
                record = j.strip().split()
                if record[0] != "#":
                    if float(record[-4]) == 317.50:
                        print(record[0])
                    if record[-4] not in series.keys():
                        series[record[-4]] = [current_replica]
                    else:
                        series[record[-4]].append(current_replica)
                    current_replica = (current_replica+1) % num

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('time series of replica exchange', fontsize=20)
        ax1.set_xlabel('time(ps)', fontsize=20)
        ax1.set_ylabel("Replica", fontsize=20)
        ax1.scatter([k for k in range(1,1001)], series["%.2f" % temperatures[0]], s=.5)

        plt.show()"""

    def crdidx(self):  # 一个温度经历的所有通道
        start = 1
        end = 3

        num = 10
        exchanges = 10000 * (end - start + 1)
        series = dict()
        for p in range(1, num + 1):
            series[str(p)] = []

        for serial in range(start, end + 1):
            path = "/Users/erik/PycharmProjects/Lacomplex/MD_WWN/4th/{0}/crdidx.dat".format(serial)
            with open(path, 'r') as f:
                for i in f.readlines():
                    record = i.strip().split()
                    if record[0][0] != "#":
                        for j in range(1, num + 1):
                            series[str(j)].append(int(record[j]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, num + 1):
            ax1 = fig.add_subplot(4, 4, k)  # 左上角第一个开始填充，从左到右
            ax1.set_title('time series of replica exchange', fontsize=2)
            ax1.set_xlabel('time(ps)', fontsize=2)
            ax1.set_ylabel("Replica", fontsize=2)
            ax1.scatter(range(1, exchanges + 1), series[str(k)], s=.1)

        plt.show()

    def repidx(self):  # 一个通道经历的所有温度
        start = 1
        end = 3

        num = 10
        exchanges = 10000 * (end - start + 1)

        series = dict()
        for p in range(1, num + 1):
            series[str(p)] = []

        for serial in range(start, end + 1):
            path = "/Users/erik/PycharmProjects/Lacomplex/MD_WWN/4th/{0}/repidx.dat".format(serial)
            with open(path, 'r') as f:
                for i in f.readlines():
                    record = i.strip().split()
                    if record[0][0] != "#":
                        for j in range(1, num + 1):
                            series[str(j)].append(int(record[j]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, num + 1):
            ax1 = fig.add_subplot(4, 4, k)  # 左上角第一个开始填充，从左到右
            ax1.set_title('time series of replica exchange', fontsize=2)
            ax1.set_xlabel('time(ps)', fontsize=2)
            ax1.set_ylabel("Replica", fontsize=2)
            ax1.scatter(range(1, exchanges+1), series[str(k)], s=.1)

        plt.show()

    def REMD_average(self):
        series = np.load("./REMD_16_SASA.npy", allow_pickle=True).item()
        score = series["2"]  # 选取第二个副本
        score_ave = []

        for i in range(1, 21):
            sum = 0
            for j in range(100*(i-1), 100*i):
                sum += score[j]
            score_ave.append(sum/100)

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('average score of replica exchange', fontsize=20)
        ax1.set_xlabel('frames', fontsize=20)
        ax1.set_ylabel("score", fontsize=20)
        ax1.bar([k for k in range(1, 21)], score_ave)
        plt.show()

    def dih_diff_feat(self):
        for_i_start = 5
        for_i_end = 5
        dih = []

        for i in range(for_i_start, for_i_end + 1):
            path = "/Users/erik/Desktop/Pareto/forward/{0}th/".format(i)
            for n in range(1, 11):
                dih_value = np.load(path + "dih_{0}.npy".format(n), allow_pickle=True)
                dih += dih_value.tolist()

        state_1 = dih[:4000]
        state_2 = dih[-4000:]

    def disdihboth(self):
        target = 'dih'
        for_i_start = 1
        for_i_end = 3
        for_dis = []
        for_dih = []
        # for_rms = []

        rev_j_start = 3
        rev_j_end = 5
        rev_dis = []
        rev_dih = []
        # rev_rms = []
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Pareto_surface', fontsize=20)
        ax1.set_xlabel('frames', fontsize=20)
        ax1.set_ylabel(target, fontsize=20)

        for i in range(for_i_start, for_i_end + 1):
            path = "/Users/erik/Desktop/Pareto/forward/{0}th/".format(i)
            # for_rms += self.read_rmsd_gmx(path + "rmsd.xvg")
            for n in range(1, 11):
                dis_value = np.load(path + "dis_{0}.npy".format(n), allow_pickle=True)
                dih_value = np.load(path + "dih_{0}.npy".format(n), allow_pickle=True)

                for_dis += dis_value.tolist()
                for_dih += dih_value.tolist()

        for j in range(rev_j_start, rev_j_end + 1):
            path = "/Users/erik/Desktop/Pareto/reverse/{0}th/".format(j)
            # rev_rms += self.read_rmsd_gmx(path + "rmsd.xvg")
            for n in range(1, 11):
                dis_value = np.load(path + "dis_{0}.npy".format(n), allow_pickle=True)
                dih_value = np.load(path + "dih_{0}.npy".format(n), allow_pickle=True)

                rev_dis += dis_value.tolist()
                rev_dih += dih_value.tolist()

        for_dis, for_dih = self.resemble_average(for_dis, for_dih)
        rev_dis, rev_dih = self.resemble_average(rev_dis, rev_dih)
        if target == "dih":
            ax1.scatter(range(len(rev_dih)), rev_dih, s=.5)
            ax1.scatter(range(len(for_dih)), for_dih, s=.5)
        if target == "dis":
            ax1.scatter(range(len(rev_dis)), rev_dis, s=.5)
            ax1.scatter(range(len(for_dis)), for_dis, s=.5)

        plt.show()
    def show_weigths(self):
        w_dis = np.load("./w_dis.npy", allow_pickle=True)
        w_dih = np.load("./w_dih.npy", allow_pickle=True)
        aa_contact = list(np.load("./aa_contact.npy", allow_pickle=True).item())

        for i in np.where(np.abs(w_dis)>20)[0]:
            print(aa_contact[i])

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('', fontsize=20)
        ax1.set_xlabel('feat_num', fontsize=20)
        ax1.set_ylabel('weights', fontsize=20)
        ax1.scatter(range(len(w_dih)), w_dih, s=5)
        # plt.show()
    def output_ATOMS(self):
        a_name, b_name = self.readHeavyAtom("./md1.pdb",monitor=False)[2:]
        aa_contact = list(np.load("./aa_contact.npy", allow_pickle=True).item())
        aa_contact.sort()
        with open("./ATOMS.dat", 'a') as file:
            for item in range(len(aa_contact)):
                input = "dist{0}: DISTANCE ATOMS={1},{2} NOPBC\n"
                for a in a_name:
                    contact = aa_contact[item][0].split('-')
                    record = a.split('-')
                    if record[0] == contact[0] and ((record[1] != "GLY" and record[2] == "CB") or ((record[1] == "GLY" and record[2] == "CA"))):
                        a_num = int(record[-1])
                for b in b_name:
                    contact = aa_contact[item][1].split('-')
                    record = b.split('-')
                    if record[0] == contact[0] and ((record[1] != "GLY" and record[2] == "CB") or ((record[1] == "GLY" and record[2] == "CA"))):
                        b_num = int(record[-1])
                file.write(input.format(item+1, a_num, b_num))
        file.close()

    def output_TORSION(self):
        with open("./torsion.dat", 'a') as file:
            input_A_phi_sine = "phi_sine_A{0}: TORSION ATOMS=@phi-{1} SINE NOPBC\n"
            input_A_psi_sine = "psi_sine_A{0}: TORSION ATOMS=@psi-{1} SINE NOPBC\n"
            input_B_phi_sine = "phi_sine_B{0}: TORSION ATOMS=@phi-{1} SINE NOPBC\n"
            input_B_psi_sine = "psi_sine_B{0}: TORSION ATOMS=@psi-{1} SINE NOPBC\n"

            input_A_phi_cosine = "phi_cosine_A{0}: TORSION ATOMS=@phi-{1} COSINE NOPBC\n"
            input_A_psi_cosine = "psi_cosine_A{0}: TORSION ATOMS=@psi-{1} COSINE NOPBC\n"
            input_B_phi_cosine = "phi_cosine_B{0}: TORSION ATOMS=@phi-{1} COSINE NOPBC\n"
            input_B_psi_cosine = "psi_cosine_B{0}: TORSION ATOMS=@psi-{1} COSINE NOPBC\n"
            for i in range(1, 268):
                file.write(input_A_phi_sine.format(i, i + 1))
                file.write(input_A_psi_sine.format(i + 268, i + 1))
                file.write(input_B_phi_sine.format(i + 268*2, i + 269))
                file.write(input_B_psi_sine.format(i + 268*3, i + 269))
                file.write(input_A_phi_cosine.format(i + 268 * 4, i + 1))
                file.write(input_A_psi_cosine.format(i + 268 * 5, i + 1))
                file.write(input_B_phi_cosine.format(i + 268 * 6, i + 269))
                file.write(input_B_psi_cosine.format(i + 268 * 7, i + 269))

        file.close()


    """def plotNNout(self):
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('probability', fontsize=20)
        # ax1.set_xlabel('0', fontsize=20)
        ax1.set_ylabel('probability', fontsize=20)
        ax1.set_ylabel('frame', fontsize=20)

        for i in range(1,8):
            path = './models/{0}'.format(i)
            model = tf.saved_model.load(path)
            # data_x = np.load('./iptg_nobind.npy', allow_pickle=True)
            data_x = np.load('./iptg_nobind.npy', allow_pickle=True)[500:]
            # print(data_x.shape)
            data_x = self.norm(data_x)
            data_x = tf.convert_to_tensor(data_x, dtype=tf.float32)
            out = model(data_x)
            print(out)
            ax1.plot(range(4500), out[:,1])

        # ax1.plot([0, 1], [0, 1], color='black')
        plt.show()

    def protran(self):
        result=[]
        for i in range(1,21):
            path = './models/{0}'.format(i)
            model = tf.saved_model.load(path)
            data_x = np.load('./iptg_nobind.npy', allow_pickle=True)
            data_x = self.norm(data_x)
            data_x = tf.convert_to_tensor(data_x, dtype=tf.float32)
            out = model(data_x)
            mean_model = tf.reduce_mean(out[:,0])
            result.append(mean_model)
            print(mean_model)
        print(result)
        print("total_mean:", np.mean(result))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('process', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel('probability to Nobind', fontsize=20)
        ax1.plot(range(5000),out[:,0])

        plt.show()

    def train(self,
              # i
              ):
        for i in range(7, 8):  # 批量训练神经网络
            path = self.ANN + "twostates_train.npy"  # 读取训练数据
            train_x = np.load(path, allow_pickle=True)  # 前4500是Bind，后4500是Nobind
            test_x = np.load('./iptg_nobind.npy', allow_pickle=True)  # 读取测试数据，5000

            train_y = np.zeros(shape=(train_x.shape[0]))  # 设定标签，9000
            train_y[:4500] = 1
            test_y = np.zeros(shape=(test_x.shape[0]))  # 5000

            # print(train_x.shape, test_x.shape)
            dataset_x = np.concatenate((train_x, test_x), axis=0)  # 合并训练集和测试集，14000
            # print(dataset_x.shape)

            dataset_x = self.norm(dataset_x)
            dataset_y = np.concatenate((train_y, test_y))  # 合并标签，14000

            # train
            dataset_x = tf.convert_to_tensor(dataset_x, dtype=tf.float32)
            dataset_y = tf.convert_to_tensor(dataset_y, dtype=tf.int32)
            dataset_y_onehot = tf.one_hot(dataset_y, depth=2, dtype=tf.int32)

            model = tf.keras.Sequential([  # 加个tf.就可以正常保存了！！！另外说一句，keras比tf慢了不止一点
                layers.Dense(256, activation=tf.nn.tanh),
                layers.Dense(128, activation=tf.nn.tanh),
                layers.Dense(64, activation=tf.nn.tanh),
                layers.Dense(32, activation=tf.nn.tanh),
                layers.Dense(16, activation=tf.nn.tanh),
                layers.Dense(8, activation=tf.nn.tanh),
                layers.Dense(4, activation=tf.nn.tanh),
                layers.Dense(2, activation=tf.nn.softmax)
            ])

            callbacks = MyCallback()
            model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
                          loss=tf.losses.binary_crossentropy,
                          metrics=[
                              Myacc()
                          ])
            models_path = './models/'  # saver保存路径
            logs_dir = './logs/{0}/'.format(i)
            logs_train_dir = os.path.join(logs_dir, "train")
            logs_valid_dir = os.path.join(logs_dir, "valid")

            for dir_name in [logs_dir, logs_train_dir, logs_valid_dir, models_path]:
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
            summary_writer = tf.summary.create_file_writer(logs_train_dir)

            model.fit(
                dataset_x,
                dataset_y_onehot,
                epochs=10000,
                shuffle=True,
                batch_size=100,
                validation_split=5 / 14,
                # validation_data=(dataset_x[9000:], dataset_y[9000:]),
                callbacks=[callbacks]
            )

            tf.saved_model.save(model, models_path+'{0}'.format(i))


    def testmodels(self):
        model1 = tf.saved_model.load("./modelsset2/18")
        # model2 = tf.saved_model.load("./models/2")
        # data_x = np.load('./iptg_nobind.npy', allow_pickle=True)

        data_x = np.load('./Bind.npy', allow_pickle=True)[500:]

        data_x = self.norm(data_x)
        data_x = tf.convert_to_tensor(data_x, dtype=tf.float32)

        # label = np.zeros(shape=(data_x.shape[0]))
        # label = tf.convert_to_tensor(label, dtype=tf.int32)  # 必须是int64用于计算accuracy
        out1 = model1(data_x)
        # print(out)
        # out2 = model2(data_x)
        pro1 = out1[:,1]
        # pro2 = out2[:, 0]
        # print(pro1[3754])
        # print(pro2[3754])
        print(pro1)
        print(np.where(pro1==np.min(pro1)))"""


# print(sys.argv[1])

lc = Lacomplex()
# lc.rmsd_batch()
# lc.sasa_cluster()
# lc.batchFrame2Dih()

# lc.Dihedral()
# lc.Dihedral_sc()


# lc.sasa_sf('/Users/erik/PycharmProjects/Lacomplex/SASA/sasa_md4000.pdb')
# print("$$$")
# score = lc.sasa_sf('/Users/erik/PycharmProjects/Lacomplex/MD_WWN/4th/3/after_gather/repre_310.0K/rep6_sasa')
# print(score)
# lc.check_diff('./extract_cor.txt', './md26.pdb') # 注意，若是反向的话需要使用另一边的初始结构来读取原子名称
# 0号帧
# a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = lc.readHeavyAtom(lc.frame_path+lc.frame_name.format(0))
# lc.calContact(a_atom_cor, b_atom_cor,a_atom_nam=a_atom_nam,b_atom_nam=b_atom_nam,filename=0,save_dis=True)

# lc.batchFrame2Dis()
# lc.mergeSet()
# lc.aa_contact()
# lc.ConDifPerFra(save=True)
# lc.ConDifPerFra_CB()
# lc.avedis()
# lc.statistics()
# lc.covariance()
# lc.PCA_dis()
# lc.PCA_dih()
# lc.Distance()

# lc.numperaa()
# lc.aveDistribution()
# lc.rmsf_plot()
# lc.extractFeat()

# lc.merge_dis()
# lc.merge_dih()
# lc.train()
# lc.testmodels()
# lc.plotNNout()
# lc.protran()
# lc.single_LDA_Dis("/Users/erik/PycharmProjects/Lacomplex/md3250.pdb", "extract_cor")
# lc.single_LDA_dih("/Users/erik/PycharmProjects/Lacomplex/work1/md2000.pdb")
# lc.LDA_trend()

# lc.add_chainID("./start.pdb")
# lc.relative_sasa_statistics("MET")
# a,b = lc.gather_dihedral_atom("./md1000.pdb", type="Psi")
# print(a[184])
# print(len(a))
# lc.read_extract_cor()

# lc.Pareto_surface(average=True)

# lc.process_pbc_cubic()
# lc.rmsf_plot_amber()
# lc.REMD_temperature_generation(310, 400, 8)
# lc.sasa_statistics()
# lc.Distance()
# lc.frame_path = "/home/caofan/work3/NoIptg_NoBind/frame_pbc"
# lc.batchFrame2Dih()
# lc.sasa()

# lc.Pathway()
# lc.aaNumSASA()

# lc.time_series_T()
# lc.dihdis_trend()
# lc.disdihboth()
# lc.rmsd_plot_gmx()

# lc.repidx()  # 一个通道经历的所有温度
# lc.crdidx()  # 一个温度经历的所有通道

# lc.REMD_average()
# lc.dih_diff_feat()
# lc.show_weigths()
# lc.output_ATOMS()
lc.output_TORSION()