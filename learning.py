import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)  # 取消科学计数显示


class Lacomplex:
    def __init__(self):
        # self.LenPro = 328  # protein序列长度
        self.contact_dis = 4.5  # 重原子之间的contact距离
        # self.NumContactPair = 324  # 给定contact数目

        self.startFrame = 1  # 首帧
        self.endFrame = 5 + 1  # 末帧
        self.Interval = 1  # 取帧间隔,看帧的名称

        self.dir_name = 10  # 字典存储名称

        self.Numdir = 10  # 保存的字典文件数量
        self.NumFrames = 5000 + 1  # 总帧数
        self.frame_read_path = "/home/liuhaiyan/fancao/work/consider_Tem/NoIptg_ProBindDna/md/startalloveragain_no_restrain/new/frames/"  # 存储每一帧的文件夹

        self.frame_name = "md{0}.pdb"  # 每一帧的名称
        self.csv_name = "{0}.csv"  # 每一张表的名称

        self.fig_save_path = "/Users/erik/Desktop/stat/NoIptg_ProNoBindDna/png/{0}.png"  # 图片保存路径

        self.csv_save_path = "/Users/erik/Desktop/NoIptg_ProBindDna/csv/"
        self.csv_read_path = "/Users/erik/Desktop/NoIptg_ProBindDna/csv/"
        # self.csv_save_path = "/home/liuhaiyan/fancao/work/consider_Tem/NoIptg_ProBindDna/md/startalloveragain_no_restrain/new/csv/"  # 表格保存路径
        # self.csv_read_path = "/home/liuhaiyan/fancao/work/consider_Tem/NoIptg_ProBindDna/md/startalloveragain_no_restrain/new/csv/"  # 表格读取路径

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
                record = i.strip().split()
                if len(record) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
                    break
                if record[0] != "ATOM":  # 检测ATOM起始行
                    continue
                current_chain = record[4]  # 获取chain_ID
                if record[0] == "TER" or (current_chain != chain and index):  # 检测chain_ID的变化
                    # 轨迹文件导出时没有TER行，所以由
                    # or后面条件辨别
                    index = 0
                if record[-1] != "H" and current_chain == "A":
                    xyz = [float(record[6]), float(record[7]), float(record[8])]
                    # eg: 1-N-2-LYS
                    name = record[1] + "-" + record[2] + "-" + record[5] + "-" + record[3]
                    a_atom_cor.append(xyz)
                    a_atom_nam.append(name)
                    index += 1
                    chain = current_chain
                if record[-1] != "H" and current_chain == "B":
                    xyz = [float(record[6]), float(record[7]), float(record[8])]
                    name = record[1] + "-" + record[2] + "-" + record[5] + "-" + record[3]
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
                    contact_pair.add((a_atom_nam[a], b_atom_nam[b]))  # 原子序号
                dis_array[b][a] = dis
        if save_dis:
            data_df = pd.DataFrame(dis_array)
            """
            (2469, 2469)
            0          1          2     ...       2466       2467       2468
            0     40.215690  38.799513  38.018277  ...  55.750993  56.925883  55.454399
            1     39.252392  37.844734  37.024317  ...  56.974233  58.151762  56.671630
            2     39.794514  38.400652  37.532537  ...  58.215807  59.394124  57.907032
            ...   ...        ...        ...        ...  ...        ...        ...
            2466  63.778714  63.050834  64.112389  ...  29.071550  29.093775  29.225590
            2467  64.947459  64.225658  65.292803  ...  29.094369  29.065833  29.258595
            2468  63.581168  62.847028  63.901245  ...  29.504145  29.537295  29.692566
            """
            data_df.columns = a_atom_nam  # 添加列标题
            data_df.index = b_atom_nam  # 添加索引
            """
            (2469, 2469)
                              1-N-2-LYS  5-CA-2-LYS  ...  4989-OC1-329-THR  4990-OC2-329-THR
            4991-N-2-LYS      40.215690   38.799513  ...         56.925883         55.454399
            4995-CA-2-LYS     39.252392   37.844734  ...         58.151762         56.671630
            ...               ...         ...        ...         ...               ...
            9976-C-329-THR    63.778714   63.050834  ...         29.093775         29.225590
            9977-OC1-329-THR  64.947459   64.225658  ...         29.065833         29.258595
            9978-OC2-329-THR  63.581168   62.847028  ...         29.537295         29.692566
            """
            path = self.csv_save_path + self.csv_name.format(filename)
            print("Saving:", path)
            data_df.to_csv(path, float_format="%.5f")

        return dis_array, contact_pair

    def batchFrame2Dis(self):
        total_contact = set()
        for i in range(self.startFrame, self.endFrame):
            print(len(total_contact))
            path = self.frame_read_path + self.frame_name.format(i * self.Interval)
            a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = self.readHeavyAtom(path)
            dis_array, contact_pair = self.calContact(a_atom_cor, b_atom_cor, a_atom_nam=a_atom_nam,
                                                      b_atom_nam=b_atom_nam,
                                                      filename=i * self.Interval, save_dis=True)
            total_contact = total_contact | contact_pair
            np.save(self.csv_save_path + "{0}.npy".format(self.dir_name), total_contact)
        # print(total_contact)

    def ConDifPerFra(self, save=None):
        #         列：原子对
        # 行：帧数
        contact_pair = np.load(self.csv_save_path + "total_contact.npy", allow_pickle=True).item()
        contact_pair = list(contact_pair)

        contact_dif = np.zeros(shape=(self.endFrame - self.startFrame, len(contact_pair)))
        path = self.csv_read_path + self.csv_name.format(0)
        data_df = pd.read_csv(path)
        # a_atom_nam = data_df.columns[1:]
        b_atom_nam = data_df[data_df.columns[0]]
        # a_atom_dir = {}
        # b_atom_dir = {}

        for i in range(self.startFrame, self.endFrame):  # 读取每一帧对应的原子距离csv
            path = self.csv_read_path + self.csv_name.format(i * self.Interval)  # 生成对应csv路径
            # A_atom, B_atom = self.ReadCor_CA(path)  # 导出原子坐标
            # dis_array = self.CalContact(A_atom, B_atom)[0]  # 计算原子距离矩阵
            data_df = pd.read_csv(path)
            data_df.index = b_atom_nam
            print(data_df)
            # data_array = data_df.iloc[:, 1:].values
            for j in range(len(contact_pair)):
                contact_dif[i][j] = data_df[contact_pair[j][0]][contact_pair[j][1]]
                # print(contact_dif[0:10])
        if save:
            data_df = pd.DataFrame(contact_dif)
            print(data_df)
            data_df.columns = contact_pair
            data_df.to_csv(self.csv_save_path + self.csv_name.format("contact_dif"), float_format="%.5f", )

    def readDisOutConFre(self):
        path = self.csv_read_path + self.csv_name.format(0)
        print("Reading:", path)
        data_df = pd.read_csv(path)
        frequen = np.zeros(shape=(data_df.shape[0], data_df.shape[0]))

        for i in range(self.startFrame, self.endFrame):
            path = self.csv_read_path + self.csv_name.format(i * self.Interval)
            print("Reading:", path)
            data_df = pd.read_csv(path)
            data_array = data_df.iloc[:, 1:].values
            # print(data_df["1416-CG2-95-VAL"][693])

            """             第一列      第二列                 
                        Unnamed: 0  1-N-2-LYS  ...  4989-OC1-329-THR  4990-OC2-329-THR
            0         4991-N-2-LYS   40.21569  ...          56.92588          55.45440
            1        4995-CA-2-LYS   39.25239  ...          58.15176          56.67163
            2        4997-CB-2-LYS   39.79451  ...          59.39412          57.90703
            ...                ...        ...  ...               ...               ...
            2466    9976-C-329-THR   63.77871  ...          29.09378          29.22559
            2467  9977-OC1-329-THR   64.94746  ...          29.06583          29.25860
            2468  9978-OC2-329-THR   63.58117  ...          29.53730          29.69257
            [2469 rows x 2470 columns]
            """
            # print(len(data_df.columns))  # 2470
            for a in range(data_array.shape[1]):
                # print(col)
                # print(len(col))
                for b in range(data_array.shape[0]):
                    if data_array[b][a] <= 4.5:
                        frequen[b][a] = frequen[b][a] + 1
            # print(data_df.iat[0,1])
            frequen_df = pd.DataFrame(frequen)
            frequen_df.columns = data_df.columns[1:]
            frequen_df.index = data_df[data_df.columns[0]]
            frequen_df.to_csv(self.csv_save_path + "freq{0}.csv".format(self.dir_name), float_format="%.5f")

    def mergeDir(self):
        total_contact = set()
        for i in range(1, self.Numdir + 1):
            path = self.csv_read_path + "{0}.npy".format(i)
            contact = np.load(path, allow_pickle=True).item()
            total_contact = total_contact | contact

        np.save(self.csv_save_path + "total_contact.npy", total_contact)


lc = Lacomplex()
# lc.batchFrame2Dis()
lc.readDisOutConFre()
