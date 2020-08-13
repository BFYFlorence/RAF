import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
np.set_printoptions(suppress=True)                          # 取消科学计数显示
class lacomplex:
    def __init__(self):
        self.LenPro = 328                                   # protein序列长度
        self.NumContactPair = 324                           # 给定contact数目
        self.NumFrames = 20                                 # 总帧数
        self.Interval = 100                                 # 取帧间隔
        self.path = "./frames/"                             # 存储每一帧的文件夹
        self._contact_pair = [(3, 118), (3, 141), (3, 142), (46, 139), (46, 141),
                              (47, 112), (47, 116), (47, 141), (48, 112), (48, 113),
                              (48, 114), (48, 115), (48, 116), (48, 117), (48, 118),
                              (48, 141), (49, 112), (49, 113), (49, 115), (49, 116),
                              (49, 117), (49, 118), (50, 56), (50, 113), (50, 115),
                              (50, 116), (50, 117), (50, 118), (51, 52), (51, 56),
                              (51, 113), (51, 114), (51, 115), (51, 116), (51, 117),
                              (51, 118), (52, 51), (52, 52), (52, 53), (52, 54),
                              (52, 55), (52, 56), (52, 57), (52, 116), (52, 117),
                              (52, 118), (53, 52), (53, 53), (53, 55), (53, 56),
                              (53, 57), (53, 116), (53, 117), (54, 52), (54, 56),
                              (54, 116), (54, 117), (55, 52), (55, 53), (56, 50),
                              (56, 51), (56, 52), (56, 53), (56, 56), (57, 53),
                              (70, 77), (70, 78), (70, 80), (70, 81), (70, 82),
                              (70, 84), (70, 85), (71, 71), (71, 72), (71, 74),
                              (71, 76), (71, 77), (71, 78), (71, 79), (71, 80),
                              (71, 81), (71, 82), (71, 84), (71, 85), (72, 71),
                              (72, 74), (72, 75), (72, 76), (72, 77), (72, 78),
                              (72, 79), (72, 80), (72, 81), (72, 82), (72, 293),
                              (72, 294), (73, 74), (73, 77), (73, 78), (73, 81),
                              (74, 71), (74, 72), (74, 73), (74, 74), (74, 77),
                              (74, 78), (75, 72), (76, 71), (76, 72), (76, 77),
                              (77, 70), (77, 71), (77, 72), (77, 73), (77, 74),
                              (77, 76), (77, 77), (77, 78), (77, 80), (77, 81),
                              (78, 70), (78, 71), (78, 72), (78, 73), (78, 74),
                              (78, 77), (79, 71), (79, 72), (80, 70), (80, 71),
                              (80, 72), (80, 77), (81, 69), (81, 70), (81, 71),
                              (81, 72), (81, 73), (81, 77), (81, 98), (81, 100),
                              (82, 70), (82, 71), (82, 72), (84, 70), (84, 71),
                              (84, 96), (84, 97), (84, 98), (84, 99), (84, 100),
                              (85, 70), (85, 71), (85, 98), (85, 99), (85, 100),
                              (88, 97), (88, 98), (88, 99), (93, 97), (94, 95),
                              (94, 96), (94, 97), (95, 94), (95, 95), (95, 96),
                              (95, 97), (96, 84), (96, 94), (96, 95), (96, 96),
                              (97, 84), (97, 88), (97, 93), (97, 94), (97, 95),
                              (98, 81), (98, 84), (98, 85), (98, 88), (99, 84),
                              (99, 85), (99, 88), (100, 81), (100, 84), (100, 85),
                              (112, 48), (113, 48), (113, 49), (113, 50), (113, 51),
                              (114, 51), (115, 48), (115, 49), (115, 50), (115, 51),
                              (116, 47), (116, 48), (116, 49), (116, 50), (116, 51),
                              (116, 52), (117, 48), (117, 49), (117, 50), (117, 51),
                              (117, 52), (117, 53), (118, 3), (118, 48), (118, 49),
                              (118, 50), (118, 51), (118, 52), (139, 46), (141, 3),
                              (141, 46), (142, 2), (142, 3), (221, 277), (221, 278),
                              (221, 281), (222, 277), (222, 278), (222, 279),
                              (222, 280), (222, 281), (222, 282), (223, 277),
                              (223, 278), (223, 279), (223, 280), (223, 281),
                              (224, 281), (225, 281), (226, 280), (226, 281),
                              (248, 277), (248, 278), (248, 281), (249, 278),
                              (249, 281), (251, 278), (251, 279), (251, 281),
                              (251, 282), (251, 283), (252, 278), (252, 280),
                              (252, 281), (252, 282), (252, 283), (253, 281),
                              (254, 281), (254, 282), (254, 283), (255, 280),
                              (255, 281), (255, 282), (255, 283), (255, 284),
                              (255, 285), (256, 281), (256, 283), (258, 283),
                              (258, 284), (259, 283), (259, 284), (277, 221),
                              (277, 222), (277, 223), (277, 248), (278, 221),
                              (278, 222), (278, 223), (278, 248), (278, 249),
                              (278, 251), (278, 252), (279, 222), (279, 223),
                              (279, 251), (280, 222), (280, 223), (280, 226),
                              (280, 252), (280, 255), (281, 221), (281, 222),
                              (281, 223), (281, 224), (281, 225), (281, 226),
                              (281, 248), (281, 249), (281, 251), (281, 252),
                              (281, 253), (281, 254), (281, 255), (281, 256),
                              (282, 222), (282, 251), (282, 252), (282, 254),
                              (282, 255), (282, 282), (282, 283), (283, 251),
                              (283, 252), (283, 254), (283, 255), (283, 256),
                              (283, 258), (283, 259), (283, 282), (283, 283),
                              (283, 284), (284, 255), (284, 258), (284, 259),
                              (284, 283), (285, 255), (293, 72), (294, 72)]

    def ReadCor_CA(self, path):
        A_atom = np.zeros(shape=(self.LenPro, 3))          # 残基序号-数组的索引=2
        B_atom = np.zeros(shape=(self.LenPro, 3))
        index = 0                                           # 向atom中添加A,B两条链坐标
        chain = ""
        with open(path, 'r') as f:
            for i in f.readlines():
                record = i.strip().split()
                if len(record) == 0:
                    break
                if record[0] != "ATOM":                       # 检测ATOM起始行
                    continue
                current_chain = record[4]
                if record[0] == "TER" or (current_chain != chain and index):
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
        dis_array = np.zeros(shape=(self.LenPro, self.LenPro))      # 二维距离数组，数组索引
                                                                    # 残基序号-数组的索引=2
        contact_pair = []                                           # contact数组
        for i in range(self.LenPro):
            for j in range(self.LenPro):
                x_2 = pow(a_atom[i][0] - b_atom[j][0], 2)
                y_2 = pow(a_atom[i][1] - b_atom[j][1], 2)
                z_2 = pow(a_atom[i][2] - b_atom[j][2], 2)
                dis = np.sqrt(x_2 + y_2 + z_2)
                if dis <= 10:
                    contact_pair.append((i + 2, j + 2))             # 残基序号
                dis_array[i][j] = dis
        return dis_array, contact_pair

    def ConDifPerFra(self, contact_pair):
        contact_dif = np.zeros(shape=(self.NumContactPair, self.NumFrames))
        for i in range(self.NumFrames):                             # 读取每一帧
            A_atom, B_atom = self.ReadCor_CA(self.path+"{0}ps.pdb".format(str((i+1)*self.Interval)))
            dis_array = self.CalContact(A_atom, B_atom)[0]
            for j in range(len(contact_pair)):                      # contact_dif与contact_pair的行意义相同
                                                                    # _contact_pair的顺序排列
                contact_dif[j][i] = dis_array[contact_pair[j][0]-2][contact_pair[j][1]-2]
            # print(contact_dif[0:10])
        return contact_dif

    def ConStat(self, contact_dif):
        # 0.min 1.max 2.diff 3.ave 4.var 5.pair1 6.pair2
        constat = np.zeros(shape=(self.NumContactPair,7))
        for i in range(self.NumContactPair):                        # 读取每一个_contact_pair
            record = contact_dif[i]
            constat[i][0] = np.min(record)
            constat[i][1] = np.max(record)
            constat[i][2] = constat[i][1] - constat[i][0]
            constat[i][3] = np.average(record)
            constat[i][4] = np.var(record)
            constat[i][5] = self._contact_pair[i][0]
            constat[i][6] = self._contact_pair[i][1]
        return constat

    def ConDisTrend(self, contact_dif, pos):
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('ps', fontsize=20)
        ax1.set_ylabel("dis", fontsize=20)
        colors = list(mcolors.CSS4_COLORS.keys()) * 3  # 颜色变化
        for i in range(self.NumContactPair):  # 按每个contact_pair绘图
            print(i)
            if self._contact_pair[i][0] in pos:
                ax1.plot(range(self.Interval, self.Interval * (self.NumFrames + 1), self.Interval), contact_dif[i],
                                 marker='o', color=mcolors.CSS4_COLORS[colors[i]])
        plt.show()

    def ConSituPos_Var(self, constat):
        # 0.min 1.max 2.diff 3.ave 4.var 5.pair1 6.pair2
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure', fontsize=20)
        ax1.set_xlabel('pos', fontsize=20)
        ax1.set_ylabel("Var", fontsize=20)
        plt.xticks(range(0, 330, 10), fontsize=12)
        contact_var_sort = constat[constat[:, 6].argsort()]
        plt.bar(x=contact_var_sort[:, -1], height=contact_var_sort[:, 4])

        plt.show()