import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import datetime
import tensorboard
import os
import re

np.set_printoptions(suppress=True)  # 取消科学计数显示
# np.set_printoptions(threshold=np.inf)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Lacomplex:
    def __init__(self):
        self.contact_dis = 4.5  # 重原子之间的contact距离

        self.startFrame = 0  # 首帧                         batchFrame2Dis       ConDifPerFra    extractFeat
        self.endFrame = 0 + 1  # 末帧                        batchFrame2Dis                ConDifPerFra   extractFeat
        self.set_name = 1  # 字典存储名称, 用于set和freq的存储    batchFrame2Dis

        # batchFrame2Dis     mergeSet    ConDifPerFra    averageDis    numperaa     aveDistribution
        # extractFeat
        self.csv_path = "./"  # 表格读取路径
        self.frame_path = "./"  # 存储每一帧的文件夹             batchFrame2Dis

        self.startSet = 1  # 字典起始                           mergeSet
        self.endSet = 1+1  # 字典终止                           mergeSet


        self.Interval = 1  # 取帧间隔,看帧的名称
        self.Numset = 10  # 保存的集合文件数量
        self.train_data_name = "Nobind"                      # extractFeat
                                                        # batchFrame2Dis

        self.frame_name = "md{0}.pdb"  # 每一帧的名称
        self.csv_name = "{0}.csv"  # 每一张表的名称

        self.fig_save_path = "/Users/erik/Desktop/stat/NoIptg_ProNoBindDna/png/{0}.png"  # 图片保存路径

        self.vmd_rmsd_path = "/Users/erik/Desktop/work/consider_tem/NoIptg_ProNoBindDna/"    # rmsd
        self.rmsd_name = "restrain.dat"     # rmsd

        # self.csv_save_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProNoBindDna/md/csv/"
        # self.csv_read_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProNoBindDna/md/csv/"
        # self.csv_save_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProBindDna/md/csv/"  # 表格保存路径

        # extractFeat
        self.NNread = "./"


    @staticmethod
    def readHeavyAtom(path) -> np.array:
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
                resName = lc.processIon(record[17:20].strip())  # PRO
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

    def mergeSet(self):
        total_contact = set()
        for i in range(self.startSet, self.endSet):
            path = self.csv_path + "{0}.npy".format(i)
            contact = np.load(path, allow_pickle=True).item()
            total_contact = total_contact | contact
        print(total_contact)
        np.save(self.csv_path + "total_contact.npy", total_contact)

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
                contact_dif[i-self.startFrame][j] = data_df[contact_pair[j][0]][contact_pair[j][1]]  # 先列再行

        if save:
            data_df = pd.DataFrame(contact_dif)
            data_df.columns = contact_pair
            data_df.to_csv(self.csv_path + self.csv_name.format("contact_dif"), float_format="%.5f", )

    def averageDis(self):
        path = self.csv_path + self.csv_name.format("contact_dif")  # contact_dif
        contact_dif = pd.read_csv(path)
        ave_array = np.zeros(shape=(len(contact_dif.columns)-1,1))
        for i in range(1, len(contact_dif.columns)):
            ave_array[i-1][0] = np.mean(np.array(contact_dif[contact_dif.columns[i]]))
        ave_df = pd.DataFrame(ave_array)
        ave_df.columns = ["ave"]
        ave_df.index = contact_dif.columns[1:]
        #                                                   average_dif
        ave_df.to_csv(self.csv_path + self.csv_name.format("average_dif"), float_format="%.5f", )

    def numperaa(self):        # average_dif
        path = self.csv_path + "whole_cp.npy"
        cp = list(np.load(path, allow_pickle=True).item())
        print(len(cp))
        freq_array = [0]*(329-62+1)
        for i in cp:
            record = i[1].split('-')            # 0，1代表画A链还是B链
            pos = int(record[0]) - 62
            freq_array[pos] = freq_array[pos] + 1

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('pos', fontsize=20)
        ax1.set_ylabel("freq", fontsize=20)
        ax1.plot(range(62, 330), freq_array)  #       ###############!!!!!!!
        plt.show()

    def aveDistribution(self):                          # average_dif
        path = self.csv_path + self.csv_name.format("average_dif")
        average_dif = pd.read_csv(path)
        average_dif.index = average_dif[average_dif.columns[0]]
        data = []
        patern = re.compile("'(.*?)'")
        for cp in average_dif.index:
            result = re.findall(patern, cp)
            # if result[0].split('-')[2] == 'CB' and result[1].split('-')[2] == 'CB': # 只取CB
            data.append(average_dif[average_dif.columns[1]][cp])
        plt.hist(data, bins=50, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.show()

    def rmsd(self):     # 单位为Å
        path = self.vmd_rmsd_path + self.rmsd_name
        frame = []
        rms = []
        with open(path, 'r') as file:
            for i in file.readlines():
                record = i.split()
                if len(record) != 0 and record[0] not in ['frame','0']:
                    print(record)
                    frame.append(float(record[0]))
                    rms.append(float(record[1]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel("RMSD(Å)", fontsize=20)
        ax1.plot(frame,rms)
        plt.show()

    def processIon(self, aa):   # 处理质子化条件
        if aa in ['ASP', 'ASH']:
            return 'ASP'
        if aa in ['HIS', 'HIE', 'HID', 'HIP']:
            return 'HIS'
        return aa

    def extractFeat(self):
        train_data = []
        # contact_li都是原始状态的残基名称，不再有ASH、HID等质子化名称 whole_cp
        contact_li = list(np.load(self.NNread + "total_contact.npy", allow_pickle=True).item())
        contact_li.sort()  # 保证contact_pair顺序一致
        # patern = re.compile("'(.*?)'")
        # print(contact_li)

        for i in range(self.startFrame, self.endFrame):
            csv_path = self.csv_path + self.csv_name.format(i * self.Interval)
            print("Reading",csv_path)
            dis_df = pd.read_csv(csv_path)
            print(dis_df)
            row = dis_df[dis_df.columns[0]]
            dis_df.index = row
            # dis_df.columns = new_col

            feat = []
            for cp in contact_li:
                # print(cp)
                # if cp[0][:3] != '109': and cp[1][:3] != '109':   # 两个初始结构在109位的残基不同
                # if cp[0].split('-')[2]=='CB' and cp[1].split('-')[2]=='CB':
                feat.append(dis_df[cp[0]][cp[1]])   # 先列后行
            train_data.append(feat)  # 文件csv和数组索引差1
        train_array_x = np.array(train_data)
        # print(train_array_x)
        np.save(self.NNread + "{0}.npy".format(self.train_data_name), train_array_x)

    def norm(self, data):
        # min-max标准化
        min_val = np.min(data)
        max_val = np.max(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - min_val) / (max_val - min_val)
        return data

    def mergetrain(self):
        bind = np.load("./Bind.npy", allow_pickle=True)
        nobind = np.load("./Nobind.npy", allow_pickle=True)
        train = np.vstack((bind, nobind))
        print(bind, nobind, train)
        np.save("./train.npy", train)


    def train(self, train_bool=None):
        path = self.NNread + "CP_ALL/train.npy"
        data = np.load(path, allow_pickle=True)    # 前1500是Bind，后1500是Nobind, (3000, 9394)
        # print(data[0])

        label = np.zeros(shape=(data.shape[0]))
        label[:4500] = 1
        # print(label.shape)
        # train
        train_x = np.vstack((data[:4000],data[4500:8500]))
        lc.norm(train_x)

        train_y = np.hstack((label[:4000],label[4500:8500]))

        train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
        train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)

        train_y = tf.one_hot(train_y, depth=2)
        # print(train_x)
        # print(train_y)

        # validation
        valid_x = np.vstack((data[4000:4500], data[8500:]))
        valid_y = np.hstack((label[4000:4500], label[8500:]))

        if train_bool:
            # model
            # 3800+ features
            model = keras.Sequential([
                layers.Dense(1024, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(8, activation='relu'),
                layers.Dense(2)
            ])
            log_dir = "/Users/erik/PycharmProjects/Lacomplex/train_log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            summary_writer = tf.summary.create_file_writer(log_dir)
            optimizer = optimizers.Adam(learning_rate=0.00001)

            train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32).shuffle(32)
            itrator = iter(train)

            for i in range(5000000):
                with tf.GradientTape() as tape:
                    train = next(itrator)

                    x = train[0]
                    y = train[1]

                    out = model(x)
                    loss = tf.square(out-y)
                    loss = tf.reduce_sum(loss) / x.shape[0]
                    print("loss:", loss)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                with summary_writer.as_default():       # 显示tensorboard必须
                    tf.summary.scalar("loss", float(loss), step=i)
                model.save("/Users/erik/PycharmProjects/Lacomplex/model.5h")
        else:
            # print(valid_x[0])
            model = keras.models.load_model("./CP_ALL/model.5h")
            out = model(valid_x)
            # print(out)

            result = []
            for i in out:
                if i[0]>i[1]:
                    result.append(0)
                else:
                    result.append(1)
            print("predict:", len(result))
            print("predict:", result[250:500])
            print("ground_t:", valid_y)
            error_num = 0
            for p in range(len(result)):
                if result[p]!=valid_y[p]:
                    error_num += 1
            print(error_num/len(result))

lc = Lacomplex()
# 0号帧
# a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = lc.readHeavyAtom(lc.frame_path+lc.frame_name.format(0))
# lc.calContact(a_atom_cor, b_atom_cor,a_atom_nam=a_atom_nam,b_atom_nam=b_atom_nam,filename=0,save_dis=True)

# lc.batchFrame2Dis()
# lc.mergeSet()
# lc.ConDifPerFra(save=True)
# lc.averageDis()



# lc.numperaa()
# lc.aveDistribution()
# lc.rmsd()
lc.extractFeat()
# lc.mergetrain()
# lc.train(train_bool=False)