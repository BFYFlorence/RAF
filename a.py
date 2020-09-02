import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)  # 取消科学计数显示


class lacomplex:
    def __init__(self):
        self.LenPro = 328  # protein序列长度
        self.contact_dis = 4.5  # 重原子之间的contact距离
        self.NumContactPair = 324  # 给定contact数目
        self.NumFrames = 5000  # 总帧数
        self.Interval = 1  # 取帧间隔,看帧的名称
        self.frame_path = "/Users/erik/PycharmProjects/frames/"  # 存储每一帧的文件夹
        self.name = "md{0}.pdb"  # 每一帧的名称
        self.fig_save_path = "/Users/erik/Desktop/stat/NoIptg_ProNoBindDna/png/{0}.png"  # 图片保存路径
        self.xlsx_save_path = "./{0}.csv"  # 表格保存路径
        self.csv_read_path = "/Users/erik/Desktop/stat/NoIptg_ProNoBindDna/constat_noheader_NoIptg_ProNoBindDna.csv"

    def ReadCor_CA(self, path):
        # 读取每个残基αC的坐标
        '''[[-21.368 108.599   3.145]
            [-19.74  109.906   6.386]
            [-19.151 113.618   6.922]
            [-16.405 114.786   4.541]
            ...
            [  8.717  80.336  46.425]
            [  7.828  76.961  48.018]
            [  8.38   74.326  45.331]
            [ 12.103  74.061  46.05 ]]'''
        print("Reading:", path)
        A_atom = np.zeros(shape=(self.LenPro, 3))  # 残基序号-数组的索引=2
        B_atom = np.zeros(shape=(self.LenPro, 3))
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
                if record[2] == "CA" and current_chain == "A":
                    A_atom[index][0] = float(record[6])
                    A_atom[index][1] = float(record[7])
                    A_atom[index][2] = float(record[8])
                    index += 1
                    chain = current_chain
                if record[2] == "CA" and current_chain == "B":
                    B_atom[index][0] = float(record[6])
                    B_atom[index][1] = float(record[7])
                    B_atom[index][2] = float(record[8])
                    index += 1
                    chain = current_chain
        return A_atom, B_atom

    def CalContact(self, a_atom, b_atom):
        # 计算a_atom, b_atom中每个原子之间的距离矩阵,
        # [328,328]
        '''[[29.82359244 30.74727287 28.20280617 ... 27.66411036]
            [30.76806536 31.93978661 29.39017023 ... 29.17042444]
            [28.22935699 29.43069216 26.80743378 ... 26.78451616]
            ...          ...         ...             ...
            [27.910641   29.4440505  26.99579821 ... 27.44559085]]'''
        dis_array = np.zeros(shape=(self.LenPro, self.LenPro))  # 二维距离数组，数组索引
        # 残基序号-数组的索引=2
        contact_pair = []  # contact数组
        for i in range(self.LenPro):
            for j in range(self.LenPro):
                x_2 = pow(a_atom[i][0] - b_atom[j][0], 2)
                y_2 = pow(a_atom[i][1] - b_atom[j][1], 2)
                z_2 = pow(a_atom[i][2] - b_atom[j][2], 2)
                dis = np.sqrt(x_2 + y_2 + z_2)
                if dis <= 10:  # 取小于等于10Å的原子对
                    contact_pair.append((i + 2, j + 2))  # 残基序号
                dis_array[i][j] = dis
        return dis_array, contact_pair

    def ConDifPerFra(self, contact_pair, save=None):
        #         列：原子对
        # 行：帧数
        contact_dif = np.zeros(shape=(self.NumFrames, self.NumContactPair))
        for i in range(self.NumFrames):  # 读取每一帧
            if i == 2917:
                file = self.frame_path + self.name.format(str((i + 1) * self.Interval))  # 生成对应帧路径
                A_atom, B_atom = self.ReadCor_CA(file)  # 导出原子坐标
                dis_array = self.CalContact(A_atom, B_atom)[0]  # 计算原子距离矩阵
                for j in range(len(contact_pair)):
                    contact_dif[i][j] = dis_array[contact_pair[j][0] - 2][contact_pair[j][1] - 2]
                # print(contact_dif[0:10])
        if save:
            '''     (3, 118)  (3, 141)  (3, 142)   ... (285, 255) (293, 72) (294, 72)
                0  14.158331  9.808241  10.138796  ...  7.835052   9.811898  9.489062
                1  15.069044  9.659956   9.482666  ...  8.357865   9.369250  9.518515
                2  12.530758  7.360065   6.848082  ...  8.642953  10.233072  9.617874
                3  11.261474  6.917341   7.184364  ...  8.467920  10.521385  9.455985
                4  12.378200  8.026156   7.560075  ...  8.272794  10.004568  9.464112'''
            data_df = pd.DataFrame(contact_dif)
            print(data_df)
            data_df.columns = self._contact_pair
            data_df.to_csv(self.xlsx_save_path.format("contact_dif"), float_format="%.5f", )
        return contact_dif

    def ConStat(self, contact_dif, save=None):
        cols = ["min", "max", "diff", "ave", "var", "pair1", "pair2"]
        constat = np.zeros(shape=(self.NumContactPair, 7))
        for i in range(self.NumContactPair):  # 读取每一个_contact_pair
            record = contact_dif[:, i]
            constat[i][0] = np.min(record)
            constat[i][1] = np.max(record)
            constat[i][2] = constat[i][1] - constat[i][0]
            constat[i][3] = np.average(record)
            constat[i][4] = np.var(record)
            constat[i][5] = self._contact_pair[i][0]
            constat[i][6] = self._contact_pair[i][1]
        if save:
            data_df = pd.DataFrame(constat)
            data_df.columns = cols
            data_df.to_csv(self.xlsx_save_path.format("constat"), float_format="%.5f")

        return constat

    def ConDisTrend(self, pos):
        contact_dif = pd.read_csv(self.csv_read_path)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('ps', fontsize=20)
        ax1.set_ylabel("dis", fontsize=20)

        # scatter_shape = ['o', '.', 'v', '^', 's', 'p', '*', '+']

        for i in range(self.NumContactPair):  # 按每个contact_pair绘图
            if self._contact_pair[i][0] in pos:
                print(self._contact_pair[i])
                ax1.plot(range(self.Interval, self.Interval * (self.NumFrames + 1), self.Interval),
                         contact_dif.values[:, i],
                         # marker=scatter_shape[i%len(scatter_shape)],
                         label=self._contact_pair[i])
        plt.legend(loc='best')
        plt.savefig(self.fig_save_path.format(pos[0]))
        # plt.show()

    def ConSituPos_Var(self):
        # 0.min 1.max 2.diff 3.ave 4.var 5.pair1 6.pair2
        constat = pd.read_csv(self.csv_read_path)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('pos', fontsize=20)
        ax1.set_ylabel("Var", fontsize=20)
        plt.xticks(range(0, 330, 10), fontsize=12)
        # contact_var_sort = constat[constat[:, 6].argsort()]
        plt.bar(x=constat.values[:, -2], height=constat.values[:, 4])

        plt.show()

    def UseCsvPlotTrend(self, pos):
        contact_dif = pd.read_csv(self.csv_read_path)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        # scatter_shape = ['o', '.', 'v', '^', 's', 'p', '*', '+']

        for i in range(self.NumContactPair):  # 按每个contact_pair绘图
            if self._contact_pair[i][0] in pos:
                print(self._contact_pair[i])
                ax1.plot(range(self.Interval, self.Interval * (self.NumFrames + 1), self.Interval),
                         contact_dif.values[:, i],
                         # marker=scatter_shape[i%len(scatter_shape)],
                         label=self._contact_pair[i])

        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('ps', fontsize=20)
        ax1.set_ylabel("dis", fontsize=20)
        plt.legend(loc='best')
        plt.savefig(self.fig_save_path.format(pos[0]))
        plt.cla()

