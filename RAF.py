import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, optimizers
from matplotlib.pyplot import MultipleLocator
import os
from collections import defaultdict
# import __main__
# __main__.pymol_argv = ['pymol', '-qc']
# import pymol as pm
import seaborn as sns
from scipy import stats

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

class RAF:
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
        self.RAF_backbone_mass = [14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01, 14.01, 12.01, 12.01]
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
        if aa in ['ASH']:
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

    def thin_data(self, li, fold=20):  # randomly
        y = []
        for i in li:
            t = np.random.uniform(low=0, high=1)
            if t < 1.0/fold:
                y.append(i)
        return y

    def space_data(self, li, interval=20):  # interval
        y = []
        count = 0
        for i in li:
            if count % interval == 0:
                y.append(i)
                count %= interval
            count += 1
        return y

    def readHeavyAtom_singleChain(self, path) -> np.array:
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
        atom_cor = []
        atom_nam = []

        with open(path, 'r') as f:
            for i in f.readlines():
                record = i.strip()
                atom = record[:4].strip()
                if atom != "ATOM":  # 检测ATOM起始行
                    continue
                # print(record)
                serial = record[6:11].strip()  # 697
                atname = record[12:16].strip()  # CA
                resName = self.processIon(record[17:20].strip())  # PRO, 已处理过质子化条件
                if resName not in self.aa:
                    continue
                resSeq = record[22:26].strip()  # 3
                cor_x = record[30:38].strip()  # Å
                cor_y = record[38:46].strip()
                cor_z = record[46:54].strip()
                element = record[13].strip()  # C

                xyz = [float(cor_x), float(cor_y), float(cor_z)]
                # eg: 2-LYS-N-697
                name = resSeq + "-" + resName + "-" + atname + "-" + serial
                if element != "H":
                    atom_cor.append(xyz)
                    atom_nam.append(name)
            return np.array(atom_cor), atom_nam

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

        result = (np.arccos(np.dot(nv1, nv2)/(np.linalg.norm(nv1)*np.linalg.norm(nv2)))/np.pi)*180*signature
        return result

    def dihedral_atom_order(self, atom_nam):
        n = 0
        atoms = ["N", "CA", "C"]
        for i in atom_nam:
            if i.split("-")[2] in atoms:
                if atoms[n % 3] != i.split("-")[2]:
                    raise Exception("二面角原子顺序错误")
                n += 1

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

    def rmsd_plot_gmx(self):  # 单位为Å
        path = "/Users/erik/Desktop/RAF/crystal_WT/test/1/"
        filename = "rmsd.xvg"
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
        ax1.scatter(range(frame), rms, s=.8)
        plt.show()
        # np.save(path+"rmsd.npy", np.array(rms))

    def gyrate_plot_gmx(self):  # 单位为Å
        # crystal_WT
        # dRafX6
        num = 5
        WD = "/Users/erik/Desktop/RAF"
        group = "crystal_WT"
        temperatures = ["300K", "344K", "384K"]
        interval = 0.02  # ns

        for temperature in temperatures:
            fig = plt.figure(num=1, figsize=(15, 8), dpi=200)
            dir_name = "/".join((WD, group, temperature, "gyrate"))
            for k in range(1, num + 1):
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                path = "/".join((WD, group, temperature, str(k), "gyrate.xvg"))
                gyrate = self.read_gyrate_gmx(path)
                average_gyrate = np.mean(gyrate)
                # print(len(rms))
                ax1 = fig.add_subplot(2, 3, k)
                ax1.cla()
                ax1.set_title(group + '_' + temperature, fontsize=20)
                ax1.set_xlabel('time(ns)', fontsize=2)
                ax1.set_ylabel("gyrate(Å)", fontsize=10)
                ax1.scatter(np.array(range(len(gyrate))) * interval, gyrate, s=.1)
                ax1.plot([0, 500], [average_gyrate, average_gyrate], color="red")
                # print(np.mean(rms))

            plt.savefig(dir_name + "/gyrate_5.png")
        # plt.legend()
        # plt.show()

    def rmsf_plot(self):  # 单位为Å
        file_name = 'rmsf_CA'
        target_file = "/Users/erik/Desktop/RAF/crystal_WT/test/1/" + file_name + ".xvg"
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
                    id = float(li[0])
                    if id == 1:
                        chain_id += 1
                    x[chain_id].append(int(id))  # -55
                    y[chain_id].append(float(li[1])*10)

        # np.save(self.vmd_rmsd_path + file_name + ".npy", np.array(y))
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('residue', fontsize=20)
        ax1.set_ylabel("RMSF(Å)", fontsize=20)

        ax1.plot(x[0], y[0])  # 4.1G
        ax1.scatter(x[0], y[0], s=20, color="green")  # 4.1G

        ax1.plot(x[1], y[1])  # NuMA
        print(x[0], x[1])
        print(y[0], y[1])


        # hydrophobic_index
        # ax1.scatter(self.hydrophobic_index[0], [y[0][i - 1] for i in self.hydrophobic_index[0]], color="black")
        # ax1.scatter(self.hydrophobic_index[1], [y[1][i - 1] for i in self.hydrophobic_index[1]], color="black")

        # hydrophilic_index
        # ax1.scatter(self.hydrophilic_index[0], y[0][self.hydrophilic_index[0] - 1], color="red")

        # either
        # ax1.scatter(self.either_index, [y[0][i - 1] for i in self.either_index], color="green")

        plt.show()

    def rmsf_plot_RAF(self):  # 单位为Å
        file_name = 'rmsf_CA'
        num = 5
        result_crystal_WT = []
        result_cry_repacking = []
        temperature = "384K"

        Strand = [[4,5,6,7,8], [15,16,17,18], [43,44,45,46,47], [72,73,74,75,76,77]]
        Alphahelix = [[25,26,27,28,29,30,31,32,33,34,35,36], [65,66,67,68]]

        for i in range(2, num+1):
            rmsf = []
            target_file = "/Users/erik/Desktop/RAF/crystal_WT/{1}/{0}/".format(i, temperature) + file_name + ".xvg"
            with open(target_file) as f:
                for j in f.readlines():
                    record = j.strip()
                    if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                        break
                    if record[0] not in ["#", "@"]:
                        li = record.split()
                        rmsf.append(float(li[1]) * 10)
            f.close()
            result_crystal_WT.append(rmsf)

        for k in range(1, num+1):
            rmsf = []
            target_file = "/Users/erik/Desktop/RAF/cry_repacking/{1}/{0}/".format(k, temperature) + file_name + ".xvg"
            with open(target_file) as f:
                for j in f.readlines():
                    record = j.strip()
                    if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                        break
                    if record[0] not in ["#", "@"]:
                        li = record.split()
                        rmsf.append(float(li[1]) * 10)
            f.close()
            result_cry_repacking.append(rmsf)

        result_crystal_WT = np.mean(np.array(result_crystal_WT), axis=0)
        result_cry_repacking = np.mean(np.array(result_cry_repacking), axis=0)

        print("crystal_WT_rmsf_mean:", np.mean(result_crystal_WT))
        print("cry_repacking_rmsf_mean:", np.mean(result_cry_repacking))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('residue_CA', fontsize=20)
        ax1.set_ylabel("RMSF(Å)", fontsize=20)

        ax1.plot(range(1, len(result_crystal_WT)+1), result_crystal_WT, color="blue")
        ax1.scatter(range(1, len(result_crystal_WT)+1), result_crystal_WT, s=20, color="blue", marker="o")

        ax1.plot(range(1, len(result_cry_repacking) + 1), result_cry_repacking, color="red")
        ax1.scatter(range(1, len(result_cry_repacking) + 1), result_cry_repacking, s=20, color="red", marker="^")

        # strand
        for strand in Strand:
            ax1.plot(strand, [0]*len(strand), color="black")
        # alpha
        for alpha in Alphahelix:
            ax1.plot(alpha, [0]*len(alpha), color="black")

        plt.show()

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

    def rmsd_between(self, path1, path2, part):
        os.system("echo 4 {2} | gmx rms -s {0} -f {1} -o rmsd_log.xvg -mw yes".format(path1, path2, part))
        with open("./rmsd_log.xvg", 'r') as file:
            for i in file.readlines():
                record = i.strip()
                if record[0] not in ["#", "@"]:
                    record = record.split()
                    rmsd = float(record[-1])*10.  # Å
        file.close()
        return rmsd

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
                if resSeq == 1 and current_aa != resName:     # 1或62，取决于是我的体系还是小红师姐的体系
                    n += 1
                record = record[:21] + chain_ID[n] + record[22:]
                current_aa = resName
                print(record)

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

    def rmsd_plot_amber(self):
        path = "/Users/erik/Desktop/MD_WWN/REMD/new_topo/"
        filename = "310K.data"
        frame = 0
        rms = []
        with open(path + filename) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    rms.append(float(li[1]))  # Å already Å
                    frame += 1

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel("RMSD(Å)", fontsize=20)
        ax1.scatter(range(frame), rms, s=.1)
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
        end = 1

        num = 8  # num of temperatures
        exchanges = 100000 * (end - start + 1)
        series = dict()
        for p in range(1, num + 1):
            series[str(p)] = []

        for serial in range(start, end + 1):
            path = "/Users/erik/Desktop/MD_WWN/REMD/new_topo/crdidx.dat".format(serial)
            with open(path, 'r') as f:
                for i in f.readlines():
                    record = i.strip().split()
                    if record[0][0] != "#":
                        for j in range(1, num + 1):
                            series[str(j)].append(int(record[j]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, num + 1):
            ax1 = fig.add_subplot(3, 3, k)  # 左上角第一个开始填充，从左到右
            ax1.set_title('time series of replica exchange', fontsize=2)
            ax1.set_xlabel('time(ps)', fontsize=2)
            ax1.set_ylabel("Replica", fontsize=2)
            ax1.scatter(range(1, exchanges + 1), series[str(k)], s=.1)

        plt.show()

    def repidx(self):  # 一个通道经历的所有温度
        start = 1
        end = 1

        num = 8
        exchanges = 100000 * (end - start + 1)

        series = dict()
        for p in range(1, num + 1):
            series[str(p)] = []

        for serial in range(start, end + 1):
            path = "/Users/erik/Desktop/MD_WWN/REMD/new_topo/repidx.dat".format(serial)
            with open(path, 'r') as f:
                for i in f.readlines():
                    record = i.strip().split()
                    if record[0][0] != "#":
                        for j in range(1, num + 1):
                            series[str(j)].append(int(record[j]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, num + 1):
            ax1 = fig.add_subplot(3, 3, k)  # 左上角第一个开始填充，从左到右
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

    def recover_pdb(self, pdb_path, serial):
        path = pdb_path.format(serial)
        recover_pdb = []
        start = 61
        current_chain = 'A'
        with open(path, 'r') as file:
            for i in file.readlines():
                record = i.strip()
                if record[:4] == 'ATOM':
                    name = record[12:16].strip()
                    chain_ID = record[21]
                    # resSeq = record[22:26]
                    # print(chain_ID, resSeq)
                    # print(record)
                    if chain_ID == current_chain:
                        if name == 'N':
                            start += 1
                    else:
                        current_chain  =chain_ID
                        start = 61
                        if name == 'N':
                            start += 1
                    record = record[:22] + "{:>4s}".format(str(start)) + record[26:] + '\n'
                    recover_pdb.append(record)
        file.close()

        write_pdb = pdb_path.format(str(serial)+"_recover")
        # print(write_pdb)

        with open(write_pdb, 'a') as f:
            for k in recover_pdb:
                f.write(k)
        f.close()

    def cal_COM(self, cors:np.array, mass:list):
        assert len(cors) == len(mass)
        M = np.sum(mass)
        add_mass = []
        for i in range(len(mass)):
            add_mass.append(cors[i]*mass[i])

        add_mass = np.sum(np.array(add_mass), axis=0)
        print(add_mass/M)

    def output_COM_restraint(self):
        atoms = ["N", "CA", "C"]
        len_chain_A = 57
        len_total = 74
        # for i in range(len_chain_A+1, len_total+1):
        #     print(i,",", i,",", i,",",end="", sep="")
        for i in range(1, (len_total-len_chain_A)*3+1):
            print("grnam2({0})='{1}'".format(i, atoms[(i-1)%3]),",", sep="", end="")

    def Kdist_plot(self):
        start = 1
        end = 6

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)  # 左上角第一个开始填充，从左到右
        for k in range(start, end + 1):
            path = "/Users/erik/Desktop/MD_WWN/REMD/new_topo/"

            filename = "Kdist.{0}.dat"
            distance = []
            with open(path + filename.format(k)) as f:
                for j in f.readlines():
                    record = j.strip()
                    if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                        break
                    if record[0] not in ["#", "@"]:
                        li = record.split()
                        distance.append(float(li[1]))  # Å already Å

            ax1.set_title('dbscan kdist', fontsize=2)
            ax1.set_xlabel('frames', fontsize=2)
            ax1.set_ylabel("Distance", fontsize=2)

            ax1.plot(range(len(distance)), distance, label=str(k))

        plt.legend()
        plt.show()

    def cnumvtime(self):
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)  # 左上角第一个开始填充，从左到右
        path = "/Users/erik/Desktop/MD_WWN/REMD/new_topo/repre_310.0K/cnumvtime.dat"
        cnum = []
        with open(path) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    cnum.append(float(li[1]))
        plt.ylim((-5, 20))
        ax1.set_title('cluster', fontsize=2)
        ax1.set_xlabel('frames', fontsize=2)
        ax1.set_ylabel("cnum", fontsize=2)
        ax1.scatter(range(len(cnum)), cnum, s=.1)
        plt.show()

    def find_lowest_ESeq(self):
        path = "/Users/erik/Desktop/RAF/designlog"
        E = []
        with open(path, 'r') as file:
            for i in file.readlines():
                record = i.strip().split()
                E.append(float(record[-2]))

        E = np.array(E)
        print(np.argsort(E))  # 升序
        print(E[0])
        print(E[99])

    def hbond_plot_gmx(self):
        num = 5
        WD = "/Users/erik/Desktop/RAF"
        group1 = "crystal_WT"
        group2 = "dRafX6"
        temperatures = ["300K", "344K", "384K"]
        filenames = ["hbond_SS", "hbond_SM", "hbond_SW", "hbond_MW"]

        paths = ["/".join((WD, "HBOND")),
                 "/".join((WD, "HBOND", "{0}_vs_{1}".format(group1, group2)))]
        for dir_name in paths:
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

        for temperature in temperatures:
            fig, ax_arr = plt.subplots(2, 2, figsize=(15, 8))
            for filename in range(len(filenames)):
                hbondgroup1 = []
                hbondgroup2 = []
                for k in range(1, num+1):
                    pathgroup1 = "/".join((WD, group1, temperature, str(k)))
                    pathgroup2 = "/".join((WD, group2, temperature, str(k)))

                    pathgroup1 = pathgroup1 + "/{0}.xvg".format(filenames[filename])
                    pathgroup2 = pathgroup2 + "/{0}.xvg".format(filenames[filename])
                    # print(pathgroup1)
                    # print(pathgroup2)

                    with open(pathgroup1) as file:
                        for j in file.readlines():
                            record = j.strip()
                            if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                                break
                            if record[0] not in ["#", "@"]:
                                li = record.split()
                                hbondgroup1.append(int(li[1]))
                    file.close()

                    with open(pathgroup2) as file:
                        for j in file.readlines():
                            record = j.strip()
                            if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                                break
                            if record[0] not in ["#", "@"]:
                                li = record.split()
                                hbondgroup2.append(int(li[1]))
                    file.close()
                    # print(len(hbondgroup1))
                    # print(len(hbondgroup2))

                """ax1 = fig.add_subplot(2, 2, filename+1)
                ax1.cla()
                ax1.set_title(filenames[filename], fontsize=20)
                ax1.set_xlabel('hbond_num', fontsize=2)
                ax1.set_ylabel("freq", fontsize=2)"""
                sns.kdeplot(np.array(hbondgroup1), shade=True, color="blue", bw_method=.3, ax=ax_arr[filename//2][filename%2]).set_title(filenames[filename])
                sns.kdeplot(np.array(hbondgroup2), shade=True, color="red", bw_method=.3, ax=ax_arr[filename//2][filename%2])
            # plt.show()
            plt.savefig("/".join((WD, "HBOND", "{0}_vs_{1}".format(group1, group2), temperature+".png")))

        # ax1.hist(hbond_cry_repacking, bins=100, color="green")
        # ax1.hist(hbond_crystal_WT, bins=100, color="blue")

    def gather_dihedral_atom_singChain(self, path, type=None):
        result = []
        atoms = ["CA", "N", "C"]
        atom_cor, atom_nam = self.readHeavyAtom_singleChain(path)
        ang_atom_cor = []

        self.dihedral_atom_order(atom_nam)

        for k in range(len(atom_nam)):
            if atom_nam[k].split("-")[2] in atoms:  # 把组成二面角原子的坐标加入
                ang_atom_cor.append(atom_cor[k])

        if type == "Phi":
            ang_atom_cor.reverse()

        ang_atom_cor = np.array(ang_atom_cor)
        ang_residue = []

        for m in range(1, int(ang_atom_cor.shape[0] / 3) + 1):
            ang_residue.append(ang_atom_cor[3 * (m - 1):3 * m + 1])

        for q in ang_residue:
            result.append(self.dihedral(q[0], q[1], q[2], q[3]) if q.shape[0] == 4 else 360)

        if type == "Phi":
            result.reverse()

        return result

    def rmsd_plot_gmx_inter(self):  # 单位为Å
        # crystal_WT
        # dRafX6
        num = 2
        WD = "/Users/erik/Desktop/RAF"
        group = "crystal_WT"
        temperatures = ["384K"]
        interval = 0.02  # ns

        for temperature in temperatures:
            fig = plt.figure(num=1, figsize=(15, 8), dpi=200)
            dir_name = "/".join((WD, group, temperature, "RMSD"))
            print(temperature+":")
            list_ave = []
            for k in range(11, 13):
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                path = "/".join((WD, group, temperature, str(k), "rmsd_500ns.xvg"))
                print(path)
                rms = self.read_rmsd_gmx(path)
                average_rms = np.mean(rms)
                list_ave.append(average_rms)
                print(average_rms, len(rms))
                ax1 = fig.add_subplot(2, 3, k-10)
                ax1.cla()
                ax1.set_title(group+'_'+temperature, fontsize=20)
                ax1.set_xlabel('time(ns)', fontsize=2)
                ax1.set_ylabel("Backbone RMSD(Å)", fontsize=10)
                ax1.scatter(np.array(range(len(rms)))*interval, rms, s=.1)
                ax1.plot([0, 500], [average_rms, average_rms], color="red")
            print("ave:", np.mean(list_ave))
            plt.savefig(dir_name+"/rmsd_5.png")
        # plt.legend()
        # plt.show()

    def rmsd_plot_gmx_intra(self):  # 单位为Å
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        target = 1  # the serial of simulation
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('Time(ns)', fontsize=20)
        ax1.set_ylabel("Backbone RMSD(Å)", fontsize=20)
        interval = 0.02  # ns

        path_1 = "/Users/erik/Desktop/RAF/crystal_WT/test/{0}/".format(target)
        path_2 = "/Users/erik/Desktop/RAF/cry_repacking/test/{0}/".format(target)

        filename = "rmsd.xvg"
        frame = 0
        rms_1 = []
        time_1 = []
        with open(path_1+filename) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    rms_1.append(float(li[1])*10)  # Å
                    time_1.append(frame*interval)
                    frame += 1

        frame = 0
        rms_2 = []
        time_2 = []
        with open(path_2 + filename) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    rms_2.append(float(li[1]) * 10)  # Å
                    time_2.append(frame * interval)
                    frame += 1

        ax1.scatter(time_1, rms_1, s=.8, label="crystal_WT")
        ax1.scatter(time_2, rms_2, s=.8, label="cry_repacking")
        plt.legend()
        plt.show()

    def dih_RAF(self):
        crystal_WT_phi = []
        crystal_WT_psi = []
        cry_repacking_phi = []
        cry_repacking_psi = []
        serial = 5
        num = 10
        target = "psi"
        temperature = "test"

        # 残基序号
        Strand = [[4, 5, 6, 7, 8], [15, 16, 17, 18], [43, 44, 45, 46, 47], [72, 73, 74, 75, 76, 77]]
        Alphahelix = [[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], [65, 66, 67, 68]]
        WT_seq =        ['THR', 'SER', 'ASN', 'THR', 'ILE', 'ARG', 'VAL', 'PHE', 'LEU', 'PRO', 'ASN', 'LYS', 'GLN', 'ARG', 'THR', 'VAL', 'VAL', 'ASN', 'VAL', 'ARG', 'ASN', 'GLY', 'MET', 'SER', 'LEU', 'HIS', 'ASP', 'CYS', 'LEU', 'MET', 'LYS', 'ALA', 'LEU', 'LYS', 'VAL', 'ARG', 'GLY', 'LEU', 'GLN', 'PRO', 'GLU', 'CYS', 'CYS', 'ALA', 'VAL', 'PHE', 'ARG', 'LEU', 'LEU', 'HIS', 'GLU', 'HIS', 'LYS', 'GLY', 'LYS', 'LYS', 'ALA', 'ARG', 'LEU', 'ASP', 'TRP', 'ASN', 'THR', 'ASP', 'ALA', 'ALA', 'SER', 'LEU', 'ILE', 'GLY', 'GLU', 'GLU', 'LEU', 'GLN', 'VAL', 'ASP', 'PHE', 'LEU']

        repacking_seq = ['ALA', 'ASP', 'ARG', 'THR', 'ILE', 'GLU', 'VAL', 'GLU', 'LEU', 'PRO', 'ASN', 'LYS', 'GLN', 'ARG', 'THR', 'VAL', 'ILE', 'ASN', 'VAL', 'ARG', 'PRO', 'GLY', 'LEU', 'THR', 'LEU', 'LYS', 'GLU', 'ALA', 'LEU', 'LYS', 'LYS', 'ALA', 'LEU', 'LYS', 'VAL', 'ARG', 'GLY', 'ILE', 'ASP', 'PRO', 'ASN', 'LYS', 'VAL', 'GLN', 'VAL', 'TYR', 'LEU', 'LEU', 'LEU', 'SER', 'GLY', 'ASP', 'ASP', 'GLY', 'ALA', 'GLU', 'GLN', 'PRO', 'LEU', 'SER', 'LEU', 'ASN', 'HIS', 'PRO', 'ALA', 'GLU', 'ARG', 'LEU', 'ILE', 'GLY', 'LYS', 'LYS', 'LEU', 'LYS', 'VAL', 'VAL', 'PRO', 'LEU']

        for k in range(1, serial+1):
            for i in range(1, num+1):
                sample = np.load("/Users/erik/Desktop/RAF/crystal_WT/{2}/{1}/phi_{0}.npy".format(i, k, temperature), allow_pickle=True)
                crystal_WT_phi += sample.tolist()
            for i in range(1, num+1):
                sample = np.load("/Users/erik/Desktop/RAF/crystal_WT/{2}/{1}/psi_{0}.npy".format(i, k, temperature), allow_pickle=True)
                crystal_WT_psi += sample.tolist()
            for i in range(1, num+1):
                sample = np.load("/Users/erik/Desktop/RAF/cry_repacking/{2}/{1}/phi_{0}.npy".format(i, k, temperature), allow_pickle=True)
                cry_repacking_phi += sample.tolist()
            for i in range(1, num+1):
                sample = np.load("/Users/erik/Desktop/RAF/cry_repacking/{2}/{1}/psi_{0}.npy".format(i, k, temperature), allow_pickle=True)
                cry_repacking_psi += sample.tolist()

        seqlen = len(WT_seq)
        samplelen = len(crystal_WT_phi)
        print(seqlen, samplelen)

        crystal_WT_phi = np.array(crystal_WT_phi)
        crystal_WT_psi = np.array(crystal_WT_psi)
        cry_repacking_phi = np.array(cry_repacking_phi)
        cry_repacking_psi = np.array(cry_repacking_psi)

        # print(np.std(crystal_WT_psi[:,0]))

        crystal_WT_phi_mean = []
        crystal_WT_psi_mean = []
        cry_repacking_phi_mean = []
        cry_repacking_psi_mean = []

        crystal_WT_phi_std = []
        crystal_WT_psi_std = []
        cry_repacking_phi_std = []
        cry_repacking_psi_std = []

        # test
        """i = 11
        temp_WT = crystal_WT_phi[:, i]
        temp_repack = cry_repacking_phi[:, i]

        mode_WT = stats.mode(temp_WT.astype(np.int64))[0][0]
        mode_repack = stats.mode(temp_repack.astype(np.int64))[0][0]

        diff_WT = temp_WT - mode_WT
        diff_repack = temp_repack - mode_repack

        for p in range(samplelen):
            if diff_WT[p] < -180:
                diff_WT[p] = 360 + diff_WT[p]
            if diff_WT[p] > 180:
                diff_WT[p] = diff_WT[p] - 360

        for p in range(samplelen):
            if diff_repack[p] < -180:
                diff_repack[p] = 360 + diff_repack[p]
            if diff_repack[p] > 180:
                diff_repack[p] = diff_repack[p] - 360

        temp_WT = diff_WT + mode_WT
        temp_repack = diff_repack + mode_repack

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('residues', fontsize=20)
        ax1.set_ylabel("Dihedral", fontsize=20)
        ax1.hist(temp_WT, bins=100)"""





        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('residues', fontsize=20)
        ax1.set_ylabel("Dihedral", fontsize=20)

        if target == "psi":
            # psi
            for i in range(seqlen):
                temp_WT = crystal_WT_psi[:, i]
                temp_repack = cry_repacking_psi[:, i]

                mode_WT = stats.mode(temp_WT.astype(np.int64))[0][0]
                mode_repack = stats.mode(temp_repack.astype(np.int64))[0][0]

                diff_WT = temp_WT - mode_WT
                diff_repack = temp_repack - mode_repack

                for p in range(samplelen):
                    if diff_WT[p] < -180:
                        diff_WT[p] = 360 + diff_WT[p]
                    if diff_WT[p] > 180:
                        diff_WT[p] = diff_WT[p] - 360

                for p in range(samplelen):
                    if diff_repack[p] < -180:
                        diff_repack[p] = 360 + diff_repack[p]
                    if diff_repack[p] > 180:
                        diff_repack[p] = diff_repack[p] - 360

                temp_WT = diff_WT + mode_WT
                temp_repack = diff_repack + mode_repack

                crystal_WT_psi_mean.append(np.mean(temp_WT))
                cry_repacking_psi_mean.append(np.mean(temp_repack))
                crystal_WT_psi_std.append(np.std(temp_WT))
                cry_repacking_psi_std.append(np.std(temp_repack))

                ax1.errorbar(range(1, len(crystal_WT_psi_mean) + 1), crystal_WT_psi_mean, yerr=crystal_WT_psi_std,
                             fmt="o", color="blue")
                ax1.errorbar(range(1, len(cry_repacking_psi_mean) + 1), cry_repacking_psi_mean,
                             yerr=cry_repacking_psi_std, fmt="^", color="red", elinewidth=2)

        elif target == "phi":
            # phi
            for i in range(seqlen):
                temp_WT = crystal_WT_phi[:, i]
                temp_repack = cry_repacking_phi[:, i]

                mode_WT = stats.mode(temp_WT.astype(np.int64))[0][0]
                mode_repack = stats.mode(temp_repack.astype(np.int64))[0][0]

                diff_WT = temp_WT - mode_WT
                diff_repack = temp_repack - mode_repack

                for p in range(samplelen):
                    if diff_WT[p] < -180:
                        diff_WT[p] = 360 + diff_WT[p]
                    if diff_WT[p] > 180:
                        diff_WT[p] = diff_WT[p] - 360

                for p in range(samplelen):
                    if diff_repack[p] < -180:
                        diff_repack[p] = 360 + diff_repack[p]
                    if diff_repack[p] > 180:
                        diff_repack[p] = diff_repack[p] - 360

                temp_WT = diff_WT + mode_WT
                temp_repack = diff_repack + mode_repack

                crystal_WT_phi_mean.append(np.mean(temp_WT))
                cry_repacking_phi_mean.append(np.mean(temp_repack))
                crystal_WT_phi_std.append(np.std(temp_WT))
                cry_repacking_phi_std.append(np.std(temp_repack))

                ax1.errorbar(range(1, len(crystal_WT_phi_mean) + 1), crystal_WT_phi_mean, yerr=crystal_WT_phi_std,
                             fmt="o", color="blue", )
                ax1.errorbar(range(1, len(cry_repacking_phi_mean) + 1), cry_repacking_phi_mean,
                             yerr=cry_repacking_phi_std, fmt="^", color="red", elinewidth=2)


        """if target == "phi":
            for i in range(crystal_WT_phi.shape[-1]):
                fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)

                ax1.set_title('Phi', fontsize=20)
                ax1.set_xlabel('', fontsize=20)
                ax1.set_ylabel("{0}_{1}\nWT".format(i+1, WT_seq[i]), fontsize=20, rotation=0)

                ax2.set_title('', fontsize=20)
                ax2.set_xlabel('residues', fontsize=20)
                ax2.set_ylabel("{0}_{1}\ndRafX6".format(i+1, repacking_seq[i]), fontsize=20, rotation=0)
                # test1 = crystal_WT_phi[:,i]
                # test2 = cry_repacking_phi[:,i]
                ax1.hist(crystal_WT_phi[:,i], bins=100)
                ax2.hist(cry_repacking_phi[:,i], bins=100)
                plt.savefig("/Users/erik/Desktop/RAF/compare_WT_vs_repacking/{1}/dih/phi/{0}_phi.png".format(i + 1, temperature))
                ax1.cla()
                ax2.cla()

        elif target == "psi":
            for i in range(crystal_WT_psi.shape[-1]):
                fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)

                ax1.set_title('Psi', fontsize=20)
                ax1.set_xlabel('', fontsize=20)
                ax1.set_ylabel("{0}_{1}\nWT".format(i+1, WT_seq[i]), fontsize=15, rotation=0)

                ax2.set_title('', fontsize=20)
                ax2.set_xlabel('residues', fontsize=20)
                ax2.set_ylabel("{0}_{1}\ndRafX6".format(i+1, repacking_seq[i]), fontsize=15, rotation=0)
                # test1 = crystal_WT_phi[:,i]
                # test2 = cry_repacking_phi[:,i]
                ax1.hist(crystal_WT_psi[:,i], bins=100)
                ax2.hist(cry_repacking_psi[:,i], bins=100)
                plt.savefig("/Users/erik/Desktop/RAF/compare_WT_vs_repacking/{1}/dih/psi/{0}_psi.png".format(i+1, temperature))
                ax1.cla()
                ax2.cla()"""

        # strand
        for strand in Strand:
            ax1.plot(strand, [-220] * len(strand), color="black")
        # alpha
        for alpha in Alphahelix:
            ax1.plot(alpha, [-220] * len(alpha), color="black")

        plt.show()

    def output_aa_name(self, path):
        # 读取氨基酸序列
        print("Reading sequence:", path)
        aa_nam = []
        cur_resSeq = 0
        with open(path, 'r') as f:
            for i in f.readlines():
                record = i.strip()
                atom = record[:4].strip()
                if atom != "ATOM":  # 检测ATOM起始行
                    continue
                resName = self.processIon(record[17:20].strip())  # PRO, 已处理过质子化条件
                resSeq = int(record[22:26].strip())  # 3
                if resName not in self.aa:
                    continue
                if resSeq!=cur_resSeq:
                    aa_nam.append(resName)
                    cur_resSeq = resSeq

            return aa_nam

    def plot_PCA_2d(self):
        path = "/Users/erik/Desktop/RAF/crystal_WT/test/1/2d.xvg"
        projection_1 = []
        projection_2 = []
        with open(path) as file:
            for j in file.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    projection_1.append(float(li[0]))
                    projection_2.append(float(li[1]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Projection', fontsize=20)
        ax1.set_xlabel('Projection_1', fontsize=20)
        ax1.set_ylabel("Projection_2", fontsize=20)
        ax1.scatter(projection_1, projection_2, s=.5)
        plt.show()

    def plot_PCA_3d(self):
        path = "/Users/erik/Desktop/RAF/crystal_WT/test/1/3dproj.pdb"
        projection_1 = []
        projection_2 = []
        projection_3 = []
        with open(path, 'r') as file:
            for j in file.readlines():
                record = j.strip().split()
                if record[0]!="ATOM":
                    continue
                projection_1.append(float(record[5]))
                projection_2.append(float(record[6]))
                projection_3.append(float(record[7]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Projection', fontsize=20)
        ax1.set_xlabel('Projection_1', fontsize=20)
        ax1.set_ylabel("Projection_2", fontsize=20)
        ax1.scatter(projection_1, projection_2, s=.5)
        plt.show()

    def dih_RAF_5(self):
        WD = "/Users/erik/Desktop/RAF"
        serial = 5
        num = 10
        temperature = "384K"
        group = "dRafX6"
        paths = ['/'.join((WD, group, temperature, "dih_5")),
                 '/'.join((WD, group, temperature, "dih_5", "phi")),
                 '/'.join((WD, group, temperature, "dih_5", "psi"))]
        for dir_name in paths:
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
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
        for p in range(len(WT_seq)):  # 氨基酸位点
            fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
            for k in range(1, serial + 1):
                crystal_WT_phi = []
                for i in range(1, num + 1):
                    sample = np.load("/Users/erik/Desktop/RAF/{3}/{2}/{1}/phi_{0}.npy".format(i, k, temperature, group),
                                     allow_pickle=True).tolist()
                    crystal_WT_phi += sample

                crystal_WT_phi = np.array(crystal_WT_phi)
                ax1 = fig.add_subplot(2, 3, k)
                ax1.cla()
                ax1.set_xlabel('residues', fontsize=5)
                ax1.set_ylabel("Dihedral", fontsize=5)
                ax1.hist(crystal_WT_phi[:,p], bins=100)
            plt.savefig(
                "/Users/erik/Desktop/RAF/{2}/{1}/dih_5/phi/{0}_phi.png".format(p+1, temperature, group))

            fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
            for k in range(1, serial + 1):
                crystal_WT_psi = []
                for i in range(1, num + 1):
                    sample = np.load("/Users/erik/Desktop/RAF/{3}/{2}/{1}/psi_{0}.npy".format(i, k, temperature, group),
                                     allow_pickle=True).tolist()
                    crystal_WT_psi += sample
                crystal_WT_psi = np.array(crystal_WT_psi)
                ax1 = fig.add_subplot(2, 3, k)
                ax1.cla()
                ax1.set_xlabel('residues', fontsize=5)
                ax1.set_ylabel("Dihedral", fontsize=5)
                ax1.hist(crystal_WT_psi[:, p], bins=100)
            plt.savefig(
                "/Users/erik/Desktop/RAF/{2}/{1}/dih_5/psi/{0}_psi.png".format(p + 1, temperature, group))

        seqlen = len(WT_seq)
        samplelen = len(crystal_WT_phi)
        print(seqlen, samplelen)

    def LJ_SR_5(self):
        serial = 5
        group = "dRafX6"
        temperature = "384K"
        path = "/Users/erik/Desktop/RAF/{0}/{1}/{2}/E_LJ_SR.xvg"
        inter = 20
        dt = 0.02  # ns

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, serial+1):
            LJ_SR = []
            with open(path.format(group, temperature, k), "r") as file:
                for i in file.readlines():
                    if i[0] not in ["#","@"]:
                        record = i.strip().split()
                        LJ_SR.append(float(record[-1]))

            LJ_SR = self.space_data(li=LJ_SR, interval=inter)
            print(len(LJ_SR))
            file.close()
            ax1 = fig.add_subplot(2, 3, k)
            ax1.cla()
            ax1.set_xlabel('time(ns)', fontsize=5)
            ax1.set_ylabel("LJ_SR", fontsize=5)
            ax1.plot(np.array([l for l in range(len(LJ_SR))])*(dt*inter),
                     LJ_SR)
        plt.show()

    def E_Tem(self):
        serial = 5
        group = "dRafX6"
        temperature = "384K"
        path = "/Users/erik/Desktop/RAF/{0}/{1}/{2}/E_Tem.xvg"
        inter = 20
        dt = 0.02  # ns

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, serial+1):
            Tem = []
            with open(path.format(group, temperature, k), "r") as file:
                for i in file.readlines():
                    if i[0] not in ["#","@"]:
                        record = i.strip().split()
                        Tem.append(float(record[-1]))

            Tem = self.space_data(li=Tem, interval=inter)
            print(len(Tem))
            file.close()
            ax1 = fig.add_subplot(2, 3, k)
            ax1.cla()
            ax1.set_xlabel('time(ns)', fontsize=5)
            ax1.set_ylabel("Tem", fontsize=5)
            ax1.plot(np.array([l for l in range(len(Tem))])*(dt*inter),
                     Tem)
        plt.show()

    def E_Coulomb_SR(self):
        serial = 5
        group = "dRafX6"
        temperature = "384K"
        path = "/Users/erik/Desktop/RAF/{0}/{1}/{2}/E_Coulomb_SR.xvg"
        inter = 20
        dt = 0.02  # ns

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        for k in range(1, serial + 1):
            Coulomb_SR = []
            with open(path.format(group, temperature, k), "r") as file:
                for i in file.readlines():
                    if i[0] not in ["#", "@"]:
                        record = i.strip().split()
                        Coulomb_SR.append(float(record[-1]))

            Coulomb_SR = self.space_data(li=Coulomb_SR, interval=inter)
            print(len(Coulomb_SR))
            file.close()
            ax1 = fig.add_subplot(2, 3, k)
            ax1.cla()
            ax1.set_xlabel('time(ns)', fontsize=5)
            ax1.set_ylabel("Coulomb_SR", fontsize=5)
            ax1.plot(np.array([l for l in range(len(Coulomb_SR))]) * (dt * inter),
                     Coulomb_SR)
        plt.show()

    def construct_rmsd_matrix(self):  # used for ave.pdb for now
        part = 4  # 4 for backbone(BB); 8 for sidechain(SC) in most cases
        WD = "/home/caofan/RAF/"
        group = "crystal_WT"  # "dRafX6"
        temperatures = ["300K", "344K", "384K"]
        rmsd_mat = np.zeros(shape=(15,15), dtype=np.float32)
        paths = []
        names = []
        serial = 5
        for temperature in temperatures:
            for k in range(1, serial+1):
                paths.append(WD
                            +'{0}/{1}/{2}/'.format(group, temperature, k))
                names.append("{0}/{1}/{2}".format(group, temperature, k))

        for p in range(len(paths)):
            path1 = paths[p]
            for q in range(p+1, len(paths)):
                path2 = paths[q]
                rmsd_mat[p][q] = self.rmsd_between(path1=path1+"pro_ave.pdb", path2=path2+"pro_ave.pdb", part=part)

        print(rmsd_mat)
        np.save("./rmsd_{0}_{1}.npy".format(group, "BB" if part==4 else "SC"), rmsd_mat)

        rmsd_mat = np.load("./rmsd_{0}_{1}.npy".format(group, "BB" if part==4 else "SC"), allow_pickle=True)
        df = pd.DataFrame(rmsd_mat)
        df.columns = names
        df.index = names
        df.to_csv("./rmsd_{0}_{1}.csv".format(group, "BB" if part==4 else "SC"))

    def self_rmsd(self):
        part = 4  # 4 for backbone(BB); 8 for sidechain(SC) in most cases
        WD = "/home/caofan/RAF/"
        group = "crystal_WT"  # "dRafX6"
        temperatures = ["300K", "344K", "384K"]
        rmsd_mat = np.zeros(shape=(3, 5), dtype=np.float32)
        serial = 5
        for temperature in range(len(temperatures)):
            for k in range(1, serial + 1):
                path = WD + '{0}/{1}/{2}/'.format(group, temperatures[temperature], k)
                rmsd_mat[temperature][k-1] = self.rmsd_between(path1=path + "md{0}.tpr".format(temperatures[temperature]), path2=path + "pro_ave.pdb", part=part)

        print(rmsd_mat)
        np.save("./self_rmsd_{0}_{1}.npy".format(group, "BB" if part == 4 else "SC"), rmsd_mat)

        rmsd_mat = np.load("./self_rmsd_{0}_{1}.npy".format(group, "BB" if part == 4 else "SC"), allow_pickle=True)
        df = pd.DataFrame(rmsd_mat)
        df.columns = [k+1 for k in range(serial)]
        df.index = temperatures
        df.to_csv("./self_rmsd_{0}_{1}.csv".format(group, "BB" if part == 4 else "SC"))

    def rmsd_heatmap(self):
        # crystal_WT
        # dRafX6
        WD = "/Users/erik/Desktop/RAF/"
        dirname = "RMSD/"

        filename = "rmsd_dRafX6_SC"
        path = WD + dirname

        data = np.load(path+filename+".npy")
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('rmsd_heatmap', fontsize=20)
        ax1.set_xlabel(filename, fontsize=20)
        # ax1.set_ylabel("Projection_2", fontsize=20)
        im = ax1.imshow(data, cmap=plt.cm.hot_r)
        # 增加右侧的颜色刻度条
        plt.colorbar(im)
        # plt.show()
        plt.savefig(WD+dirname+filename+".png")

    def read_gyrate_gmx(self, path):
        # path = "/Users/erik/Desktop/Pareto/reverse/4th/"
        # filename = "gyrate.xvg"
        frame = 0
        gyrate = []
        with open(path) as f:
            for j in f.readlines():
                record = j.strip()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] not in ["#", "@"]:
                    li = record.split()
                    gyrate.append(float(li[1]) * 10)  # Å
                    frame += 1
        return gyrate

# print(sys.argv[1])
raf = RAF()
# raf.rmsd_plot_gmx()
# raf.repidx()  # 一个通道经历的所有温度
# raf.crdidx()  # 一个温度经历的所有通道
# raf.rmsd_plot_amber()
# raf.Kdist_plot()
# raf.readHeavyAtom("/Users/erik/Desktop/RAF/WT_crystal.pdb")
# raf.cnumvtime()
# raf.gyrate_plot_gmx()
raf.find_lowest_ESeq()
# raf.hbond_plot_gmx()
# cor, name = raf.readHeavyAtom_singleChain("/Users/erik/Desktop/RAF/cry_repacking/test/4/ff.pdb")
# Phi = raf.gather_dihedral_atom_singChain("/Users/erik/Desktop/RAF/cry_repacking/test/4/ff.pdb", type="Phi")
# print(Phi)
# raf.rmsd_plot_gmx_intra()
# raf.rmsd_plot_gmx_inter()
# raf.rmsd_plot_gmx()
# raf.dihdis_trend()
# raf.dih_RAF()
# print(raf.output_aa_name("/Users/erik/Desktop/RAF/crystal_WT/test/1/md15.pdb"))
# raf.rmsf_plot_RAF()
# raf.plot_PCA_2d()
# raf.plot_PCA_3d()
# raf.dih_RAF_5()
# raf.E_Coulomb_SR()
# raf.E_Tem()
# raf.LJ_SR_5()
# raf.construct_rmsd_matrix()
# raf.self_rmsd()
# raf.rmsd_heatmap()
# raf.rmsd_plot_gmx_inter()
# raf.read_rmsd_gmx()
