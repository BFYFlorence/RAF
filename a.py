import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from matplotlib.pyplot import MultipleLocator
import os
import sys
import csv

np.set_printoptions(suppress=True)  # 取消科学计数显示
# np.set_printoptions(threshold=np.inf)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Macos需要设为true
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 自定义计算acc
"""def Myaccc(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1, output_type=tf.int32),
                                               tf.argmax(y_true, axis=1, output_type=tf.int32)), tf.float32)) # 保存行数
    return accuracy
"""


class Myacc(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(tf.argmax(y_true, axis=1, output_type=tf.int32),
                                  tf.argmax(y_pred, axis=1, output_type=tf.int32)), tf.int32)
        # self.total = self.total+tf.shape(y_true)[0]
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("myacc") > 0.85:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


class Lacomplex:
    def __init__(self):
        self.contact_dis = 4.5  # 重原子之间的contact距离
        self.startFrame = 1  # 首帧                         batchFrame2Dis       ConDifPerFra    extractFeat
        self.endFrame = 5000 + 1  # 末帧                        batchFrame2Dis                ConDifPerFra   extractFeat
        self.set_name = 7  # 字典存储名称, 用于set和freq的存储    batchFrame2Dis

        # batchFrame2Dis     mergeSet    ConDifPerFra    averageDis    numperaa     aveDistribution
        # extractFeat
        self.csv_path = "./"  # 表格读取路径
        self.frame_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/Iptg_ProNoBindDna/test/md/frame/"  # 存储每一帧的文件夹             batchFrame2Dis
        self.NNread = "./"

        self.startSet = 1  # 字典起始                           mergeSet
        self.endSet = 10 + 1  # 字典终止                           mergeSet

        self.Interval = 1  # 取帧间隔,看帧的名称
        self.Numset = 10  # 保存的集合文件数量
        self.train_data_name = "iptg_nobind"  # extractFeat
        # batchFrame2Dis

        self.frame_name = "md{0}.pdb"  # 每一帧的名称
        self.csv_name = "{0}.csv"  # 每一张表的名称

        # self.fig_save_path = "/Users/erik/Desktop/stat/NoIptg_ProNoBindDna/png/{0}.png"  # 图片保存路径

        self.vmd_rmsd_path = "/Users/erik/Desktop/work/cut_DNABindDomain/Iptg_ProNoBindDna/"  # rmsd
        self.stat_name = "potential.xvg"  # rmsd

        # self.csv_save_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProNoBindDna/md/csv/"
        # self.csv_read_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProNoBindDna/md/csv/"
        # self.csv_save_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProBindDna/md/csv/"  # 表格保存路径

        # extractFeat

    def processIon(self, aa):  # 处理质子化条件
        if aa in ['ASP', 'ASH']:
            return 'ASP'
        if aa in ['HIS', 'HIE', 'HID', 'HIP']:
            return 'HIS'
        return aa

    def norm(self, data):
        # min-max标准化
        min_val = np.min(data)
        max_val = np.max(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - min_val) / (max_val - min_val)
        return data

    def mergeSet(self):
        total_contact = set()
        for i in range(self.startSet, self.endSet):
            path = self.csv_path + "{0}.npy".format(i)
            contact = np.load(path, allow_pickle=True).item()
            total_contact = total_contact | contact
        print(total_contact)
        np.save(self.csv_path + "total_contact.npy", total_contact)

    def readHeavyAtom(self, path) -> np.array:
        # 读取每条链重原子的坐标
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
        a_atom_cor = []  # 存储A链重原子坐标
        b_atom_cor = []  # 存储B链重原子坐标
        a_atom_nam = []  # 存储A链重原子名称
        b_atom_nam = []  # 存储B链重原子名称
        index = 0  # 向atom中添加A,B两条链坐标
        chain = ""
        with open(path, 'r') as f:
            for i in f.readlines():
                record = i.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break

                atom = record[:4].strip()
                if atom != "ATOM":  # 检测ATOM起始行
                    continue

                serial = record[6:11].strip()  # 697
                atname = record[12:16].strip()  # CA
                resName = self.processIon(record[17:20].strip())  # PRO
                current_chain = record[21].strip()  # 获取chain_ID,A
                resSeq = record[22:26].strip()  # 3
                cor_x = record[30:38].strip()  # Å
                cor_y = record[38:46].strip()
                cor_z = record[46:54].strip()
                element = record[76:78].strip()  # C
                if atom == "TER" or (current_chain != chain and index):  # 检测chain_ID的变化
                    # 轨迹文件导出时没有TER行，所以由
                    # or后面条件辨别
                    index = 0

                xyz = [float(cor_x), float(cor_y), float(cor_z)]
                # eg: 2-LYS-N-697
                name = resSeq + "-" + resName + "-" + atname + "-" + serial
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
        return np.array(a_atom_cor), np.array(b_atom_cor), a_atom_nam, b_atom_nam

    def calContact(self, a_atom_cor, b_atom_cor, a_atom_nam=None, b_atom_nam=None, filename=None, save_dis=None):
        # 计算a_atom, b_atom中每个原子之间的距离矩阵,
        """             a
           [[29.82359244 30.74727287 28.20280617 ... 27.66411036]
            [30.76806536 31.93978661 29.39017023 ... 29.17042444]
        b   [28.22935699 29.43069216 26.80743378 ... 26.78451616]
            ...          ...         ...             ...
            [27.910641   29.4440505  26.99579821 ... 27.44559085]]"""
        dis_array = np.zeros(shape=(a_atom_cor.shape[0], b_atom_cor.shape[0]))  # [2469,2469]
        contact_pair = set()  # contact集合
        for a in range(a_atom_cor.shape[0]):
            for b in range(b_atom_cor.shape[0]):
                x_2 = pow(a_atom_cor[a][0] - b_atom_cor[b][0], 2)
                y_2 = pow(a_atom_cor[a][1] - b_atom_cor[b][1], 2)
                z_2 = pow(a_atom_cor[a][2] - b_atom_cor[b][2], 2)
                dis = np.sqrt(x_2 + y_2 + z_2)
                if dis <= self.contact_dis:  # 取小于等于4.5Å的原子对
                    contact_pair.add((a_atom_nam[a], b_atom_nam[b]))  # 原子序号，先A后B
                dis_array[b][a] = dis
        if save_dis:
            data_df = pd.DataFrame(dis_array)
            data_df.columns = a_atom_nam  # 添加列标题
            data_df.index = b_atom_nam  # 添加索引
            path = self.csv_path + self.csv_name.format(filename)
            print("Saving:", path)
            data_df.to_csv(path, float_format="%.5f")

        return contact_pair

    def batchFrame2Dis(self):
        total_contact = set()
        path = self.csv_path + self.csv_name.format(0)
        data_df = pd.read_csv(path)
        a_atom_nam = data_df.columns[1:]
        b_atom_nam = data_df[data_df.columns[0]]

        for i in range(self.startFrame, self.endFrame):
            print(len(total_contact))
            path = self.frame_path + self.frame_name.format(i * self.Interval)
            a_atom_cor, b_atom_cor = self.readHeavyAtom(path)[:2]
            contact_pair = self.calContact(a_atom_cor, b_atom_cor, a_atom_nam=a_atom_nam,
                                           b_atom_nam=b_atom_nam,
                                           filename=i * self.Interval, save_dis=True)
            total_contact = total_contact | contact_pair

        np.save(self.csv_path + "{0}.npy".format(self.set_name), total_contact)

    def ConDifPerFra(self, save=None):
        #         列：原子对
        # 行：帧数
        contact_pair = np.load(self.csv_path + "total_contact.npy", allow_pickle=True).item()
        contact_pair = list(contact_pair)
        contact_pair.sort()  # 保证contact_pair的一致性
        contact_dif = np.zeros(shape=(self.endFrame - self.startFrame, len(contact_pair)))
        path = self.csv_path + self.csv_name.format(0)
        data_df = pd.read_csv(path)
        b_atom_nam = data_df[data_df.columns[0]]

        for i in range(self.startFrame, self.endFrame):  # 读取每一帧对应的原子距离csv
            path = self.csv_path + self.csv_name.format(i * self.Interval)  # 生成对应csv路径
            data_df = pd.read_csv(path)
            data_df.index = b_atom_nam
            for j in range(len(contact_pair)):
                contact_dif[i - self.startFrame][j] = data_df[contact_pair[j][0]][contact_pair[j][1]]  # 先列再行

        if save:
            data_df = pd.DataFrame(contact_dif)
            data_df.columns = contact_pair
            data_df.to_csv(self.csv_path + self.csv_name.format("contact_dif"), float_format="%.5f")

    def ConDifPerFra_CB(self):
        aa_contact = list(np.load(self.NNread + 'aa_contact.npy', allow_pickle=True).item())
        aa_contact.sort()
        # print(aa_contact)
        contact_dif = np.zeros(shape=(self.endFrame - self.startFrame, len(aa_contact)))
        for i in range(self.startFrame, self.endFrame):
            path = self.csv_path + self.csv_name.format(i * self.Interval)
            print("reading:", path)
            dataframe = pd.read_csv(path)
            # print(dataframe)
            new_col = ['Unnamed: 0']
            new_index = []
            for col in dataframe.columns[1:]:
                record = col.split('-')
                new_col.append(record[0] + '-' + self.processIon(record[1]) + '-' + record[2])
                # ['CB' if record[1]!='GLY' else 'CA']
            # print(new_col)
            for row in dataframe[dataframe.columns[0]]:
                record = row.split('-')
                new_index.append(record[0] + '-' + self.processIon(record[1]) + '-' + record[2])
            # print(new_index)
            dataframe.columns = new_col
            dataframe.index = new_index
            # print(dataframe)
            # print(aa_contact)

            for l in range(len(aa_contact)):
                cp = aa_contact[l]
                a_rec = cp[0].split('-')
                b_rec = cp[1].split('-')
                a_atom = a_rec[0] + '-' + self.processIon(a_rec[1]) + '-' + ['CB' if a_rec[1] != 'GLY' else 'CA'][0]
                b_atom = b_rec[0] + '-' + self.processIon(b_rec[1]) + '-' + ['CB' if b_rec[1] != 'GLY' else 'CA'][0]
                contact_dif[i - self.startFrame][l] = dataframe[a_atom][b_atom]  # 先列后行,索引需要减2

        data_df = pd.DataFrame(contact_dif)
        data_df.columns = aa_contact
        data_df.to_csv(self.csv_path + self.csv_name.format('contact_dif_CB'), float_format="%.5f")

    def avedis(self):
        path = self.csv_path + self.csv_name.format("contact_dif_CB_NoBind")  # contact_dif
        contact_dif = pd.read_csv(path)
        ave_var_array = np.zeros(shape=(len(contact_dif.columns) - 1, 2))
        for i in range(1, len(contact_dif.columns)):
            ave_var_array[i - 1][0] = np.mean(np.array(contact_dif[contact_dif.columns[i]]))
            ave_var_array[i - 1][1] = np.var(np.array(contact_dif[contact_dif.columns[i]]))
        ave_df = pd.DataFrame(ave_var_array)
        ave_df.columns = ["ave", "var"]
        ave_df.index = contact_dif.columns[1:]
        #                                                   average_dif
        ave_df.to_csv(self.csv_path + self.csv_name.format("ave_var_dif_NoBind"), float_format="%.5f")

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
        bind_path = self.csv_path + self.csv_name.format("ave_var_dif_Bind")
        nobind_path = self.csv_path + self.csv_name.format("ave_var_dif_NoBind")

        bind = pd.read_csv(bind_path)
        nobind = pd.read_csv(nobind_path)

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
        plt.show()

    def covariance(self):
        bind = np.load('./Bind.npy', allow_pickle=True)
        bind = bind[500:]
        nobind = np.load('./Nobind.npy', allow_pickle=True)
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

    def PCA(self):
        bind = np.load('./Bind.npy', allow_pickle=True)
        bind = bind[500:]
        nobind = np.load('./Nobind.npy', allow_pickle=True)
        nobind = nobind[500:]

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





    def rmsd(self):  # 单位为Å
        path = self.vmd_rmsd_path + self.rmsd_name
        frame = []
        rms = []
        with open(path, 'r') as file:
            for i in file.readlines():
                record = i.split()
                if len(record) != 0 and record[0] not in ['frame', '0']:
                    print(record)
                    frame.append(float(record[0]))
                    rms.append(float(record[1]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel("RMSD(Å)", fontsize=20)
        ax1.plot(frame, rms)
        plt.show()

    def statistics(self):
        target_file = self.vmd_rmsd_path + self.stat_name
        indicator = ['potential(kJ/mol)', 'temperature', 'pressure', 'density', 'RMSD-CA', 'RMSF(nm)'][0]
        #                   0                     1           2         3         4          5
        x = []
        y = []
        with open(target_file) as f:
            for i in f.readlines():
                i = i.strip()
                if len(i) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if i[0] != "#" and i[0] != "@":
                    li = i.split()
                    x.append(float(li[0]))
                    y.append(float(li[1]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel(indicator, fontsize=20)
        ax1.plot(range(len(x)), y)
        plt.show()

    def aa_contact(self):  # whole_cp
        path = self.csv_path + "total_contact.npy"
        cp = list(np.load(path, allow_pickle=True).item())
        cp.sort()
        aa_cp = set()
        for i in cp:
            record1 = i[0].split('-')
            record2 = i[1].split('-')
            aa_cp.add((record1[0] + '-' + record1[1], record2[0] + '-' + record2[1]))
        print(len(aa_cp))
        np.save(self.csv_path + "aa_contact.npy", aa_cp)

    def extractFeat(self):
        train_data = []
        # contact_li都是原始状态的残基名称，不再有ASH、HID等质子化名称 whole_cp
        #                                        total_contact
        contact_li = list(np.load(self.NNread + "aa_contact.npy", allow_pickle=True).item())
        contact_li.sort()  # 保证contact_pair顺序一致

        for i in range(self.startFrame, self.endFrame):
            csv_path = self.csv_path + self.csv_name.format(i * self.Interval)
            print("Reading", csv_path)
            dis_df = pd.read_csv(csv_path)

            # 删除原子序号信息，便于随机访问
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
        np.save(self.NNread + "{0}.npy".format(self.train_data_name), train_array_x)

    def mergetrain(self):
        bind = np.load("./Bind.npy", allow_pickle=True)
        nobind = np.load("./Nobind.npy", allow_pickle=True)
        train = np.vstack((bind[500:], nobind[500:]))
        # print(bind, nobind, train)
        np.save("./train.npy", train)

    def plotNNout(self):
        path = './reducedmodels/31'
        model = tf.saved_model.load(path)
        data_x = np.load('./twostates_train.npy', allow_pickle=True)[:,:10]
        # print(data_x.shape)
        data_x = self.norm(data_x)
        data_x = tf.convert_to_tensor(data_x, dtype=tf.float32)
        out = model(data_x)
        print(out)

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('probability', fontsize=20)
        ax1.set_xlabel('0', fontsize=20)
        ax1.set_ylabel('1', fontsize=20)
        ax1.scatter(out[:,0],out[:,1])
        ax1.plot([0, 1], [0, 1], color='black')
        plt.show()

    def train(self, i):
        # for i in range(2, 3):  # 批量训练神经网络
        x = []
        y = []
        path = "./iris.csv"
        with open(path, 'r') as f:
            File = csv.reader(f)
            for c in File:
                x.append([float(c[0]), float(c[1]), float(c[2]), float(c[3])])
                y.append([int(c[4])])

        x = np.array(x)
        y = np.array(y)

        dataset_x = self.norm(x)

        dataset_y = y

        # train
        dataset_x = tf.convert_to_tensor(dataset_x, dtype=tf.float32)
        dataset_y = tf.convert_to_tensor(dataset_y, dtype=tf.int32)
        dataset_y_onehot = tf.one_hot(dataset_y, depth=2, dtype=tf.int32)

        """model = tf.keras.Sequential([  # 加个tf.就可以正常保存了！！！另外说一句，keras比tf慢了不止一点
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(8, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.softmax)
        ])"""

        model = tf.keras.Sequential([  # 加个tf.就可以正常保存了！！！另外说一句，keras比tf慢了不止一点
            # layers.Dense(128, activation=tf.nn.relu),
            # layers.Dense(64, activation=tf.nn.relu),
            # layers.Dense(16, activation=tf.nn.relu),
            # layers.Dense(8, activation=tf.nn.relu),
            # layers.Dense(4, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.softmax)
        ])

        callbacks = MyCallback()
        model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
                      loss=tf.losses.binary_crossentropy,
                      metrics=[
                          Myacc()
                      ])
        models_path = './irisdmodels/'  # saver保存路径
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
            epochs=30,
            shuffle=True,
            batch_size=5,
            # validation_split=5 / 14,
            # validation_data=(dataset_x[9000:], dataset_y_onehot[9000:]),
            callbacks=[callbacks]
        )

        tf.saved_model.save(model, models_path+'{0}'.format(i))

    # 手动计算准确率
    """def calaccrate(self, model, data_x, vali, info):
        out = model(data_x)
        result = tf.argmax(out, 1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(result, vali), tf.float32))  # 保存行数
        print("calaccrate:", accuracy)"""

    def testmodels(self):
        model = tf.saved_model.load("./modeltest")

        data_x = np.load('./iptg_nobind.npy', allow_pickle=True)
        print(data_x.shape)
        data_x = self.norm(data_x)
        data_x = tf.convert_to_tensor(data_x, dtype=tf.float32)

        label = np.zeros(shape=(data_x.shape[0]))
        label = tf.convert_to_tensor(label, dtype=tf.int32)  # 必须是int64用于计算accuracy

        self.calaccrate(model, data_x, label, "error_rate:")


# print(sys.argv[1])

lc = Lacomplex()
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
# lc.PCA()

# lc.numperaa()
# lc.aveDistribution()
# lc.rmsd()
# lc.extractFeat()
# lc.mergetrain()
lc.train(sys.argv[1])
# lc.testmodels()
# lc.plotNNout()