import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets
import datetime
import tensorboard
import os
import re

np.set_printoptions(suppress=True)  # 取消科学计数显示
np.set_printoptions(threshold=np.inf)
os.environ['KMP_DUPLICATE_LIB_OK']='True'       # Macos需要设为true
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Lacomplex:
    def __init__(self):
        self.contact_dis = 4.5  # 重原子之间的contact距离

        self.startFrame = 1  # 首帧                         batchFrame2Dis       ConDifPerFra    extractFeat
        self.endFrame = 5000 + 1  # 末帧                        batchFrame2Dis                ConDifPerFra   extractFeat
        self.set_name = 7  # 字典存储名称, 用于set和freq的存储    batchFrame2Dis

        # batchFrame2Dis     mergeSet    ConDifPerFra    averageDis    numperaa     aveDistribution
        # extractFeat
        self.csv_path   = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/Iptg_ProNoBindDna/test/md/csv/"  # 表格读取路径
        self.frame_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/Iptg_ProNoBindDna/test/md/frame/"  # 存储每一帧的文件夹             batchFrame2Dis
        self.NNread     = "./"

        self.startSet = 1  # 字典起始                           mergeSet
        self.endSet = 10+1  # 字典终止                           mergeSet


        self.Interval = 1  # 取帧间隔,看帧的名称
        self.Numset = 10  # 保存的集合文件数量
        self.train_data_name = "iptg_nobind"                      # extractFeat
                                                        # batchFrame2Dis

        self.frame_name = "md{0}.pdb"  # 每一帧的名称
        self.csv_name = "{0}.csv"  # 每一张表的名称

        # self.fig_save_path = "/Users/erik/Desktop/stat/NoIptg_ProNoBindDna/png/{0}.png"  # 图片保存路径

        self.vmd_rmsd_path = "/Users/erik/Desktop/work/cut_DNABindDomain/Iptg_ProNoBindDna/"    # rmsd
        self.stat_name = "potential.xvg"     # rmsd

        # self.csv_save_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProNoBindDna/md/csv/"
        # self.csv_read_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProNoBindDna/md/csv/"
        # self.csv_save_path = "/home/liuhaiyan/fancao/work/cut_DNABindDomain/NoIptg_ProBindDna/md/csv/"  # 表格保存路径

        # extractFeat

    def processIon(self, aa):   # 处理质子化条件
        if aa in ['ASP', 'ASH']:
            return 'ASP'
        if aa in ['HIS', 'HIE', 'HID', 'HIP']:
            return 'HIS'
        return aa

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
        """path = self.csv_path + self.csv_name.format("contact_dif")  # contact_dif
        contact_dif = pd.read_csv(path)
        ave_array = np.zeros(shape=(len(contact_dif.columns)-1,1))
        for i in range(1, len(contact_dif.columns)):
            ave_array[i-1][0] = np.mean(np.array(contact_dif[contact_dif.columns[i]]))
        ave_df = pd.DataFrame(ave_array)
        ave_df.columns = ["ave"]
        ave_df.index = contact_dif.columns[1:]
        #                                                   average_dif
        ave_df.to_csv(self.csv_path + self.csv_name.format("average_dif"), float_format="%.5f", )"""
        

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

    def statistics(self):
        target_file = self.vmd_rmsd_path + self.stat_name
        indicator = ['potential(kJ/mol)', 'temperature', 'pressure', 'density', 'RMSD-CA', 'RMSF(nm)'][0]
        #                   0                     1           2         3         4          5
        x=[]
        y=[]
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
        ax1.plot(range(len(x)),y)
        plt.show()

    def aa_contact(self):       # whole_cp
        path = self.csv_path + "total_contact.npy"
        cp = list(np.load(path, allow_pickle=True).item())
        cp.sort()
        aa_cp = set()
        for i in cp:
            record1 = i[0].split('-')
            record2 = i[1].split('-')
            aa_cp.add((record1[0]+'-'+record1[1], record2[0]+'-'+record2[1]))
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
            print("Reading",csv_path)
            dis_df = pd.read_csv(csv_path)

            # 删除原子序号信息，便于随机访问
            new_col = ['Unnamed: 0']
            new_index = []
            for col in dis_df.columns[1:]:
                record = col.split('-')
                new_col.append(record[0] + '-'+record[1]+'-'+record[2])

            for index in dis_df[dis_df.columns[0]]:
                record = index.split('-')
                new_index.append(record[0]+'-'+record[1]+'-'+record[2])

            dis_df.columns = new_col
            dis_df.index = new_index

            feat = []
            for cp in contact_li:
                print(cp)

                # 因为gly没有CA，所以换为CA
                a_index = cp[0]+'-'+ ['CB' if cp[0][-3:]!='GLY' else 'CA'][0]
                b_index = cp[1]+'-'+ ['CB' if cp[1][-3:]!='GLY' else 'CA'][0]
                feat.append(dis_df[a_index][b_index])   # 先列后行
            train_data.append(feat)  # 文件csv和数组索引差1

        train_array_x = np.array(train_data)
        print(train_array_x)
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
        train = np.vstack((bind[500:], nobind[500:]))
        # print(bind, nobind, train)
        np.save("./train.npy", train)

    def calaccrate(self, model, data_x, vali, info):
        out = model(data_x)
        result = [0 if i[0] > i[1] else 1 for i in out]
        error_num = 0
        for p in range(len(result)):
            if result[p] != vali[p]:
                error_num += 1

        error_rate = error_num / len(result)
        print(info, error_rate * 100, "%")
        return error_rate

    def train(self, train_bool=None):
        for i in range(1,31):               # 批量训练神经网络
            path = self.NNread + "twostates_train.npy"
            data = np.load(path, allow_pickle=True)  # 前4500是Bind，后4500是Nobind
            # print(data[0])
            num_classes = 2
            batch_size = 20
            valid_batch_size = 128
            shuffle_buffer = 10000
            NUM_EPOCHS = 10000
            models_path = './models/{0}/'.format(i)  # saver保存路径
            logs_dir = './logs/{0}/'.format(i)
            logs_train_dir = os.path.join(logs_dir, "train")
            logs_valid_dir = os.path.join(logs_dir, "valid")


            for dir_name in [logs_dir, logs_train_dir, logs_valid_dir, models_path]:
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)

            label = np.zeros(shape=(data.shape[0]))
            label[:4500] = 1
            # train
            train_x = np.vstack((data[:4000], data[4500:8500]))
            self.norm(train_x)
            train_y = np.hstack((label[:4000], label[4500:8500]))
            train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
            train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
            train_y_onehot = tf.one_hot(train_y, depth=num_classes)
            train = tf.data.Dataset.from_tensor_slices((train_x, train_y_onehot))
            train = train.shuffle(shuffle_buffer)
            train = train.batch(batch_size)
            train = train.repeat(NUM_EPOCHS)
            iterator = iter(train)

            # validation
            valid_x = np.vstack((data[4000:4500], data[8500:]))
            valid_y = np.hstack((label[4000:4500], label[8500:]))
            valid_x = tf.convert_to_tensor(valid_x, dtype=tf.float32)
            valid_y = tf.convert_to_tensor(valid_y, dtype=tf.int32)
            valid_y_onehot = tf.one_hot(valid_y, depth=num_classes)
            valid = tf.data.Dataset.from_tensor_slices((valid_x, valid_y_onehot))
            valid = valid.batch(valid_batch_size)
            val_iterator = iter(valid)

            if train_bool:
                # model
                # 3800+ features
                # model = keras.models.load_model("./model.5h")
                model = tf.keras.Sequential([           # 加个tf.就可以正常保存了！！！另外说一句，keras比tf慢了不止一点
                    layers.Dense(128, activation='relu'),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(8, activation='relu'),
                    layers.Dense(2)
                ])
                # log_dir = "./train_log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                model.compile()

                summary_writer = tf.summary.create_file_writer(logs_train_dir)
                optimizer = optimizers.Adam(learning_rate=0.00001)
                min_error = 1.0
                step = 0
                start_time = time.time()

                while True:
                    try:
                        step += 1
                        with tf.GradientTape() as tape:
                            x,y = next(iterator)
                            out = model(x)
                            loss = tf.square(out - y)
                            loss = tf.reduce_sum(loss) / x.shape[0]
                            print("loss:", loss, "step:", step)

                        grads = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                        with summary_writer.as_default():  # 显示tensorboard必须
                            tf.summary.scalar("loss", float(loss), step=step)

                        if step%20 == 0:
                            self.calaccrate(model, train_x, train_y, "error rate training_set:")
                            error_rate = self.calaccrate(model, valid_x, valid_y, "error rate validation_set:")
                            if error_rate < min_error:
                                min_error = error_rate
                                model.save(models_path + "model_{0}".format(step))
                                # tf.saved_model.save(model, models_path + "model_{0}".format(step))
                                if min_error < 0.05:
                                    break

                    except StopIteration:
                        print("done")
                        break

            else:
                model = tf.saved_model.load("./models/model_20")
                print("loading model.......done!")
                self.calaccrate(model, valid_x, valid_y, "error rate validation_set:")

lc = Lacomplex()
# 0号帧
# a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = lc.readHeavyAtom(lc.frame_path+lc.frame_name.format(0))
# lc.calContact(a_atom_cor, b_atom_cor,a_atom_nam=a_atom_nam,b_atom_nam=b_atom_nam,filename=0,save_dis=True)

# lc.batchFrame2Dis()
# lc.mergeSet()
# lc.aa_contact()
# lc.ConDifPerFra(save=True)
# lc.averageDis()
# lc.statistics()

# lc.numperaa()
# lc.aveDistribution()
# lc.rmsd()
# lc.extractFeat()
# lc.mergetrain()
# lc.train(train_bool=True)